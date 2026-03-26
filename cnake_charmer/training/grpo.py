"""
GRPO training using TRL GRPOTrainer with environment_factory.

TRL handles the multi-turn rollout loop internally:
generate → detect tool calls → execute environment methods → append results → repeat.
We define the environment class and reward function; TRL does the rest.
"""

import json
import logging

from datasets import Dataset

from cnake_charmer.sources.base import ProblemSpec
from cnake_charmer.training.environment import CythonToolEnvironment
from cnake_charmer.training.prompts import format_user_prompt

logger = logging.getLogger(__name__)


def cython_reward(environments, **kwargs):
    """Reward function that scores the final Cython code from each environment.

    Called by TRL after each rollout completes. Accesses the environment's
    accumulated state to compute a composite reward.
    """
    rewards = []
    for env in environments:
        try:
            score = env.get_composite_score()
            rewards.append(score)
        except Exception as e:
            logger.warning(f"Reward computation failed: {e}")
            rewards.append(0.0)
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
    num_generations: int = 4,
    max_completion_length: int = 2048,
    max_tool_calling_iterations: int = 3,
    use_vllm: bool = False,
    learning_rate: float = 1e-6,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    logging_steps: int = 1,
    save_steps: int = 50,
    report_to: str = "tensorboard",
    peft_config: dict | None = None,
    **extra_config,
):
    """Create a TRL GRPOTrainer configured for Cython code generation.

    Args:
        model: HuggingFace model name or path.
        problems: List of ProblemSpecs to train on (builds dataset automatically).
        dataset: Pre-built dataset (alternative to problems).
        output_dir: Where to save checkpoints and logs.
        num_generations: Number of rollouts per prompt (G in GRPO).
        max_completion_length: Max tokens per completion (across all turns).
        max_tool_calling_iterations: Max tool-calling turns per rollout.
        use_vllm: Use vLLM for faster generation (requires trl[vllm]).
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
        use_vllm=use_vllm,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to=report_to,
        bf16=True,
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
