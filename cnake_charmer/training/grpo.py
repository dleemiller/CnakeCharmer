"""
GRPO training using TRL GRPOTrainer with environment_factory.

Two-component graduated reward design:
  R_total = λ * R_atomic + (1-λ) * R_progress + R_bonus

R_atomic:   Average per-call quality across all tool calls in the trajectory.
R_progress: Delta-clipped step-over-step improvement (rewards progress, penalizes regression).
R_bonus:    Small bonuses for completion, efficiency; penalties for format errors.

Staged curriculum shifts R_atomic weights from correctness-heavy to performance-heavy
at the training midpoint.
"""

import json
import logging

from datasets import Dataset

from cnake_charmer.sources.base import ProblemSpec
from cnake_charmer.training.environment import CythonToolEnvironment
from cnake_charmer.training.prompts import format_user_prompt

logger = logging.getLogger(__name__)

# Staged curriculum weights for R_atomic
STAGE1_WEIGHTS = {
    "correctness": 0.40,
    "performance": 0.10,
    "annotations": 0.20,
    "lint": 0.05,
    "memory_safety": 0.10,
}

STAGE2_WEIGHTS = {
    "correctness": 0.25,
    "performance": 0.30,
    "annotations": 0.20,
    "lint": 0.05,
    "memory_safety": 0.10,
}

# R_total mixing coefficient
LAMBDA = 0.7  # R_atomic weight; (1-LAMBDA) for R_progress


def _get_curriculum_weights(trainer_state) -> dict:
    """Get R_atomic weights based on training progress."""
    if trainer_state is None or trainer_state.max_steps <= 0:
        return STAGE1_WEIGHTS

    progress = trainer_state.global_step / trainer_state.max_steps
    if progress < 0.5:
        return STAGE1_WEIGHTS
    return STAGE2_WEIGHTS


def cython_reward(environments, trainer_state=None, log_extra=None, log_metric=None, **kwargs):
    """Graduated reward: R_atomic + R_progress + R_bonus.

    Called by TRL after each rollout completes. Uses per-step scores
    tracked by the environment during tool-calling iterations.
    """
    weights = _get_curriculum_weights(trainer_state)
    rewards = []

    # Per-env metrics for logging
    speedups = []
    correctnesses = []
    annotations = []
    compile_count = 0
    correct_count = 0
    tool_call_counts = []

    for env in environments:
        try:
            r_atomic = env._get_atomic_reward(weights)
            r_progress = env._get_progress_reward()
            r_bonus = env._get_bonus_reward()

            r_total = LAMBDA * r_atomic + (1 - LAMBDA) * r_progress + r_bonus
            # Clamp to [0, 1] — negative total shouldn't happen often but be safe
            r_total = max(0.0, min(1.0, r_total))
            rewards.append(r_total)

            # Collect per-env metrics
            if env.step_scores:
                last = env.step_scores[-1]
                speedups.append(last.get("speedup", 0.0))
                correctnesses.append(last.get("correctness", 0.0))
                annotations.append(last.get("annotations", 0.0))
                if last.get("compiled"):
                    compile_count += 1
                if last.get("correctness", 0) >= 1.0:
                    correct_count += 1
            tool_call_counts.append(env.num_tool_calls)

        except Exception as e:
            logger.warning(f"Reward computation failed: {e}")
            rewards.append(0.0)
            speedups.append(0.0)
            correctnesses.append(0.0)
            annotations.append(0.0)
            tool_call_counts.append(0)

    n = len(environments) or 1

    # Log per-completion columns
    if log_extra:
        log_extra("speedup", speedups)
        log_extra("correctness", correctnesses)
        log_extra("annotation", annotations)
        log_extra("tool_calls", tool_call_counts)

    # Log aggregate scalar metrics
    if log_metric:
        log_metric("compile_rate", compile_count / n)
        log_metric("correct_rate", correct_count / n)
        log_metric("mean_speedup", sum(speedups) / n)
        log_metric("mean_annotation", sum(annotations) / n)
        log_metric("mean_tool_calls", sum(tool_call_counts) / n)
        # Track curriculum stage
        if trainer_state and trainer_state.max_steps > 0:
            log_metric(
                "curriculum_stage",
                2.0 if trainer_state.global_step / trainer_state.max_steps >= 0.5 else 1.0,
            )

    return rewards


def build_dataset(problems: list[ProblemSpec]) -> Dataset:
    """Build a HuggingFace Dataset from ProblemSpecs for TRL GRPOTrainer.

    Each row contains the prompt (as chat messages) plus metadata fields
    that get passed to the environment's reset() as kwargs.
    """
    prompts = []
    python_codes = []
    func_names = []
    test_cases_list = []
    benchmark_args_list = []

    for p in problems:
        prompts.append(
            [{"role": "user", "content": format_user_prompt(p.python_code, p.description)}]
        )
        python_codes.append(p.python_code)
        func_names.append(p.func_name)
        test_cases_list.append(json.dumps(p.test_cases))
        benchmark_args_list.append(json.dumps(p.benchmark_args))

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "python_code": python_codes,
            "func_name": func_names,
            "test_cases": test_cases_list,
            "benchmark_args": benchmark_args_list,
        }
    )


def create_trainer(
    model: str = "Qwen/Qwen3-0.6B",
    problems: list[ProblemSpec] | None = None,
    dataset: Dataset | None = None,
    output_dir: str = "./output",
    num_generations: int = 8,
    max_completion_length: int = 8192,
    max_tool_calling_iterations: int = 5,
    learning_rate: float = 5e-7,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 3,
    logging_steps: int = 1,
    save_steps: int = 25,
    report_to: str = "tensorboard",
    peft_config: dict | None = None,
    **extra_config,
):
    """Create a TRL GRPOTrainer configured for Cython code generation.

    Uses CISPO loss, HF generate (no vLLM), graduated reward function,
    and CythonToolEnvironment for multi-turn tool calling.

    Args:
        model: HuggingFace model name or path.
        problems: List of ProblemSpecs to train on (builds dataset automatically).
        dataset: Pre-built dataset (alternative to problems).
        output_dir: Where to save checkpoints and logs.
        num_generations: Number of rollouts per prompt (G in GRPO).
        max_completion_length: Max tokens per completion (across all turns).
        max_tool_calling_iterations: Max tool-calling turns per rollout.
        learning_rate: Learning rate for training.
        peft_config: Optional LoRA/PEFT config dict.
    """
    from trl import GRPOConfig, GRPOTrainer

    if dataset is None:
        if problems is None:
            raise ValueError("Either problems or dataset must be provided")
        dataset = build_dataset(problems)

    config = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_tool_calling_iterations=max_tool_calling_iterations,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to=report_to,
        bf16=True,
        gradient_checkpointing=True,
        # CISPO loss — proven on gpt-oss-20b by Chroma Context-1
        loss_type="cispo",
        scale_rewards="batch",
        beta=0.001,  # small KL penalty prevents length blowup
        **extra_config,
    )

    trainer_kwargs = {
        "model": model,
        "args": config,
        "train_dataset": dataset,
        "reward_funcs": cython_reward,
        "environment_factory": CythonToolEnvironment,
    }

    if peft_config:
        from peft import LoraConfig

        trainer_kwargs["peft_config"] = LoraConfig(**peft_config)

    return GRPOTrainer(**trainer_kwargs)
