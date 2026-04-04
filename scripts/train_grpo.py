"""
GRPO training for gpt-oss-20b-cython-sft on Python→Cython optimization.

Reinforcement learning on top of SFT using TRL GRPOTrainer with:
  - CISPO loss (proven on gpt-oss-20b by Chroma Context-1)
  - Graduated reward: R_atomic + R_progress + R_bonus
  - Staged curriculum: correctness-first → performance push
  - CythonToolEnvironment for multi-turn tool calling
  - HF generate (no vLLM — single GPU memory constraint)
  - adamw_8bit optimizer to save memory

Usage:
    # Dry run (2 problems, 1 step)
    uv run --no-sync python scripts/train_grpo.py --max-steps 1 --n-problems 2

    # Full training
    uv run --no-sync python scripts/train_grpo.py

    # Custom model
    uv run --no-sync python scripts/train_grpo.py --model models/gpt-oss-20b-cython-sft-merged
"""

import argparse
import logging
import random
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Monkey-patch TRL for Harmony chat template compatibility.
#
# TRL's GRPOTrainer requires two things for tool-calling:
#   1. tokenizer.response_schema — regex-based parser for tool calls
#   2. get_training_chat_template() — prefix-preserving template
#
# The Harmony format uses <|return|> for final assistant message vs <|end|>
# for mid-conversation, breaking prefix preservation by 1 token.
# We bypass the check since TRL re-tokenizes prompts from scratch at each
# tool-calling iteration anyway.
# ---------------------------------------------------------------------------
import trl.chat_template_utils as _trl_chat_utils
import trl.trainer.grpo_trainer as _trl_grpo
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from cnake_charmer.training.environment import CythonToolEnvironment
from cnake_charmer.training.grpo import build_dataset, cython_reward
from cnake_data.loader import discover_pairs

_orig_add_response_schema = _trl_chat_utils.add_response_schema
_orig_get_training_template = _trl_chat_utils.get_training_chat_template


def _patched_add_response_schema(tokenizer):
    """Skip add_response_schema — we set it manually on the tokenizer."""
    if getattr(tokenizer, "response_schema", None):
        return tokenizer
    return _orig_add_response_schema(tokenizer)


def _patched_get_training_chat_template(tokenizer):
    """Return None for Harmony format — accept the <|return|> vs <|end|> mismatch.

    The 1-token difference (<|return|> at end-of-conversation vs <|end|> mid-conversation)
    doesn't affect the tool-calling loop because TRL re-tokenizes prompts from scratch at
    each iteration. The mismatch only means the final token of a completed turn differs,
    which is harmless since that token is never part of a generation prefix.
    """
    if getattr(tokenizer, "response_schema", None) and _is_harmony_template(tokenizer):
        return None  # Use native template as-is
    return _orig_get_training_template(tokenizer)


def _is_harmony_template(tokenizer):
    """Detect if this tokenizer uses the Harmony chat template."""
    ct = getattr(tokenizer, "chat_template", "") or ""
    return "<|start|>" in ct and "<|channel|>" in ct and "<|call|>" in ct


def _harmony_parse_tool_calls(text: str) -> dict:
    """Parse Harmony-format assistant output into TRL-compatible tool call dict.

    Handles:
      - Analysis channel: <|start|>assistant<|channel|>analysis<|message|>...<|end|>
      - Tool calls: <|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{JSON}<|call|>
      - Plain content without tool calls
    """
    import json as _json
    import re as _re

    reasoning = ""
    # Match analysis channel — may or may not have <|start|>assistant prefix
    # (TRL strips the prefix since it's part of the generation prompt)
    m = _re.search(
        r"(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>",
        text,
        _re.DOTALL,
    )
    if m:
        reasoning = m.group(1).strip()

    # Only recognize tools that the environment actually exposes
    _KNOWN_TOOLS = {"evaluate_cython"}

    tool_calls = []
    for m in _re.finditer(
        r"(?:<\|start\|>)?assistant to=functions\.(\w+)"
        r"<\|channel\|>commentary json<\|message\|>"
        r"(\{.+?\})<\|call\|>",
        text,
        _re.DOTALL,
    ):
        name = m.group(1)
        if name not in _KNOWN_TOOLS:
            logging.getLogger(__name__).warning(f"Ignoring hallucinated tool call: {name}")
            continue
        try:
            args = _json.loads(m.group(2))
        except _json.JSONDecodeError:
            args = {"_raw": m.group(2)}
        tool_calls.append({"type": "function", "function": {"name": name, "arguments": args}})

    result = {"role": "assistant", "content": reasoning}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


_orig_parse_response = _trl_chat_utils.parse_response


def _patched_parse_response(tokenizer, ids):
    """Use Harmony parser for gpt-oss tokenizers, original for everything else."""
    if _is_harmony_template(tokenizer):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        parsed = _harmony_parse_tool_calls(text)
        if "tool_calls" in parsed:
            _trl_chat_utils._validate_tool_calls(parsed["tool_calls"])
            logger.info(
                f"Harmony parser found {len(parsed['tool_calls'])} tool call(s): "
                f"{[tc['function']['name'] for tc in parsed['tool_calls']]}"
            )
        return parsed
    return _orig_parse_response(tokenizer, ids)


def _patched_get_tool_suffix_ids(self, tool_messages):
    """Harmony-compatible tool suffix ID computation.

    In Harmony format, tool responses appear as:
      <|start|>functions.NAME to=assistant<|channel|>commentary<|message|>CONTENT<|end|>
    followed by the generation prompt:
      <|start|>assistant

    We directly construct these tokens rather than using apply_chat_template,
    which has strict message ordering requirements.
    """
    tok = self.processing_class
    parts = []
    for msg in tool_messages:
        name = msg.get("name", "evaluate_cython")
        content = msg.get("content", "")
        # Harmony tool response format
        tool_response = (
            f"<|start|>functions.{name} to=assistant"
            f"<|channel|>commentary<|message|>{content}<|end|>"
        )
        parts.append(tool_response)
    # Generation prompt for next assistant turn
    parts.append("<|start|>assistant")

    suffix_text = "".join(parts)
    return tok.encode(suffix_text, add_special_tokens=False)


# Apply patches
_trl_chat_utils.add_response_schema = _patched_add_response_schema
_trl_chat_utils.get_training_chat_template = _patched_get_training_chat_template
_trl_chat_utils.parse_response = _patched_parse_response
# Also patch the imports in grpo_trainer module
_trl_grpo.add_response_schema = _patched_add_response_schema
_trl_grpo.get_training_chat_template = _patched_get_training_chat_template
_trl_grpo.parse_response = _patched_parse_response
# Patch _get_tool_suffix_ids on the GRPOTrainer class
_trl_grpo.GRPOTrainer._get_tool_suffix_ids = _patched_get_tool_suffix_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "models/gpt-oss-20b-cython-sft-merged"
DEFAULT_OUTPUT = "models/gpt-oss-20b-cython-grpo"


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for Cython optimization")

    # Model
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or HF ID")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")

    # Data
    parser.add_argument(
        "--n-problems",
        type=int,
        default=None,
        help="Limit to N random problems (for testing). Default: all.",
    )
    parser.add_argument(
        "--eval-holdout",
        type=int,
        default=50,
        help="Number of problems to hold out for evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # GRPO
    parser.add_argument("--num-generations", type=int, default=8, help="Rollouts per prompt (G)")
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=8192,
        help="Max completion tokens across all turns (prompt is ~850 tokens on top)",
    )
    parser.add_argument(
        "--max-tool-iters",
        type=int,
        default=5,
        help="Max tool-calling iterations per rollout",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train batch size (must be divisible by num-generations)",
    )
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max training steps (-1 = use epochs)"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Loss
    parser.add_argument(
        "--loss-type",
        default="cispo",
        choices=["grpo", "cispo", "dapo", "vespo", "sapo"],
        help="GRPO loss variant",
    )
    parser.add_argument("--beta", type=float, default=0.001, help="KL penalty coefficient")

    # Generation
    parser.add_argument(
        "--cache-impl",
        default="static",
        choices=["static", "dynamic", None],
        help="KV cache implementation (static pre-allocates like vLLM, dynamic grows)",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    # Logging
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--report-to", default="tensorboard")

    return parser.parse_args()


def load_model(model_path: str):
    """Load model with flex_attention + liger kernels for memory efficiency.

    Mirrors the SFT training setup from train_sft.py.
    """
    logger.info(f"Loading model: {model_path}")

    # Use eager attention — gpt-oss only supports eager and flex_attention.
    # flex_attention causes tensor shape mismatch with gradient_checkpointing
    # during the GRPO generate→train loop (shape changes between forward passes).
    # Eager + liger kernels is the safe combo for GRPO.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto",
    )
    logger.info("Loaded with eager attention")

    # Apply liger kernels for memory-efficient non-attention ops
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss

        apply_liger_kernel_to_gpt_oss(model, fused_linear_cross_entropy=False)
        logger.info("Applied liger kernels")
    except ImportError:
        logger.warning("liger-kernel not available, skipping")
    except Exception as e:
        logger.warning(f"liger kernels failed: {e}, continuing without")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set a marker response_schema so TRL knows tool-call parsing is configured.
    # Actual parsing is handled by our monkey-patched parse_response (see top of file).
    tokenizer.response_schema = {"_harmony": True}
    logger.info("Set Harmony response schema marker (parsing handled by monkey-patch)")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU after model load: {allocated:.1f}GB")

    return model, tokenizer


def load_problems(args):
    """Load and split problems into train/eval sets."""
    logger.info("Discovering problems...")
    all_problems = discover_pairs()
    logger.info(f"Found {len(all_problems)} problems")

    random.seed(args.seed)
    shuffled = list(all_problems)
    random.shuffle(shuffled)

    # Hold out eval problems
    eval_problems = shuffled[: args.eval_holdout]
    train_problems = shuffled[args.eval_holdout :]

    if args.n_problems:
        train_problems = train_problems[: args.n_problems]

    logger.info(f"Train: {len(train_problems)} problems | Eval: {len(eval_problems)} problems")
    return train_problems, eval_problems


def main():
    args = parse_args()
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Loss type: {args.loss_type}")

    # GPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        logger.info(f"GPU: {props.name} ({total_gb:.0f} GB)")

    # Load problems and build dataset
    train_problems, eval_problems = load_problems(args)
    train_dataset = build_dataset(train_problems)
    logger.info(f"Train dataset: {len(train_dataset)} rows")

    # Save eval problem IDs for later
    eval_ids = [p.problem_id for p in eval_problems]
    eval_path = Path(args.output_dir) / "eval_problem_ids.txt"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text("\n".join(eval_ids))
    logger.info(f"Saved {len(eval_ids)} eval problem IDs to {eval_path}")

    # GRPOConfig
    config = GRPOConfig(
        output_dir=args.output_dir,
        # GRPO params
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_iters,
        # Loss
        loss_type=args.loss_type,
        scale_rewards="batch",
        beta=args.beta,
        # Training
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        # Memory optimization
        optim="adamw_8bit",
        gradient_checkpointing=True,
        bf16=True,
        # Generation — static cache pre-allocates KV like vLLM, avoids reallocation
        cache_implementation=args.cache_impl,
        temperature=args.temperature,
        # Logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model with flex_attention + liger kernels
    model, tokenizer = load_model(args.model)

    logger.info("Creating GRPOTrainer...")
    logger.info(f"  num_generations={args.num_generations}")
    logger.info(f"  max_completion_length={args.max_completion_length}")
    logger.info(f"  max_tool_calling_iterations={args.max_tool_iters}")
    logger.info(f"  batch_size={args.batch_size} x grad_accum={args.grad_accum}")
    logger.info(f"  lr={args.lr}, loss={args.loss_type}, beta={args.beta}")
    logger.info(f"  LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info("  optimizer=adamw_8bit")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=cython_reward,
        environment_factory=CythonToolEnvironment,
        peft_config=lora_config,
    )

    # Override auto-generated tool schemas with exact SFT schemas from data/tools.json.
    # TRL auto-generates schemas from environment method signatures via get_json_schema(),
    # but the SFT tools.json has explicit "default" fields that the Harmony template
    # renders as "// default: true" comments. Without this, the rendered tool section
    # differs from what the model was trained on.
    from cnake_charmer.training.prompts import get_tools

    sft_tools = get_tools()
    if sft_tools:
        trainer.tools = sft_tools
        logger.info(f"Overrode tool schemas with {len(sft_tools)} SFT tools from data/tools.json")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"GPU after trainer init: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
        )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # Save final model
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
