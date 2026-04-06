"""
GEPA prompt optimization for Cython agent, per teacher model.

Evolves the ReAct agent's instructions using GEPA's reflective
optimization with our composite reward + textual feedback.

Run separately for each teacher model to get model-specific prompts.

Usage:
    # Optimize for gpt-oss-120b (local vLLM)
    uv run python scripts/optimize_prompt.py \
        --model openrouter/openai/gpt-oss-120b \
        --budget light --subset 30

    # Optimize for DeepSeek V3.2 (OpenRouter)
    OPENROUTER_API_KEY=... uv run python scripts/optimize_prompt.py \
        --model openrouter/deepseek/deepseek-v3.2 \
        --budget medium --subset 30

    # List saved optimized prompts
    uv run python scripts/optimize_prompt.py --list
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.traces.lm import CythonReActAgent, model_slug
from cnake_charmer.training.dspy_agent import (
    cython_metric,
    problem_to_example,
)
from cnake_data.loader import discover_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "optimized_prompts"


def run_optimization(
    model_id: str,
    api_key: str,
    base_url: str | None = None,
    reflection_model: str | None = None,
    budget: str = "light",
    subset: int = 150,
    val_size: int = 50,
    difficulty: str | None = None,
    max_iters: int = 5,
    num_threads: int = 2,
    reflection_minibatch_size: int = 3,
    temperature: float = 0.7,
    extra_body: dict | None = None,
    use_thinking: bool = False,
):
    """Run GEPA optimization for a specific model.

    Uses dual-model architecture per GEPA best practices:
    - Student LM (model_id): fast/cheap model doing inference (99% of calls)
    - Reflection LM (reflection_model): stronger model for error analysis (1% of calls)
    """

    # Configure student LM
    lm_kwargs = {"api_key": api_key, "temperature": temperature}
    # Only set api_base for local models — OpenRouter models route via litellm
    if base_url and not model_id.startswith("openrouter/"):
        lm_kwargs["api_base"] = base_url
    if extra_body:
        lm_kwargs["extra_body"] = extra_body

    lm = dspy.LM(model_id, **lm_kwargs)
    dspy.settings.configure(lm=lm)
    logger.info(f"Student LM: {model_id}")

    # Configure reflection LM (stronger model for analyzing errors)
    # GEPA requires a reflection LM — if none specified, reuse the student model.
    # Reflection model may be remote (OpenRouter) even when student is local.
    reflection_model_id = reflection_model or model_id
    if reflection_model:
        # Remote reflection model — needs OpenRouter API key
        reflection_api_key = os.environ.get("APIKEY", os.environ.get("OPENROUTER_API_KEY", ""))
        reflection_lm = dspy.LM(
            reflection_model_id,
            api_key=reflection_api_key,
            temperature=1.0,
            max_tokens=32000,
        )
    else:
        # Same as student (local)
        reflection_kwargs = {"api_key": api_key, "temperature": 1.0, "max_tokens": 32000}
        if base_url:
            reflection_kwargs["api_base"] = base_url
        reflection_lm = dspy.LM(reflection_model_id, **reflection_kwargs)
    if reflection_model:
        logger.info(f"Reflection LM: {reflection_model}")
    else:
        logger.info(f"Reflection LM: {model_id} (same as student)")

    # Load problems
    problems = discover_pairs()
    if difficulty:
        problems = [p for p in problems if p.difficulty == difficulty]
        logger.info(f"Filtered to {len(problems)} {difficulty} problems")

    # Sample for optimization (GEPA doesn't need the full dataset)
    problems = problems[:subset]
    logger.info(f"Using {len(problems)} problems for optimization")

    # Convert to DSPy examples
    examples = [problem_to_example(p) for p in problems]

    # Split train/val with explicit val size
    actual_val = min(val_size, len(examples) // 2)
    trainset = examples[actual_val:]
    valset = examples[:actual_val]
    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Create agent
    if use_thinking:
        logger.info("Using ThinkingReAct (native LM thinking mode)")
    agent = CythonReActAgent(max_iters=max_iters, use_thinking=use_thinking)

    # Output dir for this model
    slug = model_slug(model_id)
    save_dir = PROMPTS_DIR / slug
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run GEPA
    logger.info(f"Starting GEPA optimization (budget={budget})...")
    # Checkpoint dir for incremental saves and resume
    log_dir = save_dir / "gepa_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # File-based graceful stop: `touch <save_dir>/gepa.stop` to end early
    stop_file = save_dir / "gepa.stop"
    if stop_file.exists():
        stop_file.unlink()  # clear stale stop file from previous run

    from gepa.utils.stop_condition import FileStopper

    gepa_kwargs = {
        "metric": cython_metric,
        "auto": budget,
        "num_threads": num_threads,
        "reflection_minibatch_size": reflection_minibatch_size,
        "track_stats": True,
        "add_format_failure_as_feedback": True,
        "warn_on_score_mismatch": False,
        "log_dir": str(log_dir),
        "gepa_kwargs": {
            "stop_callbacks": FileStopper(str(stop_file)),
        },
    }
    logger.info(f"To stop gracefully: touch {stop_file}")
    if reflection_lm:
        gepa_kwargs["reflection_lm"] = reflection_lm

    optimizer = dspy.GEPA(**gepa_kwargs)

    optimized = optimizer.compile(
        student=agent,
        trainset=trainset,
        valset=valset if valset else None,
    )

    # Save optimized program
    optimized.save(save_dir / "program.json")

    # Save metadata
    metadata = {
        "model": model_id,
        "reflection_model": reflection_model,
        "budget": budget,
        "subset": subset,
        "difficulty": difficulty,
        "max_iters": max_iters,
        "reflection_minibatch_size": reflection_minibatch_size,
        "num_problems_train": len(trainset),
        "num_problems_val": len(valset),
        "val_size": val_size,
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save detailed stats if available
    if hasattr(optimized, "detailed_results"):
        try:
            stats = optimized.detailed_results.to_dict()
            (save_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Could not save detailed stats: {e}")

    logger.info(f"Saved optimized program to {save_dir}")
    return optimized, save_dir


def list_saved_prompts():
    """List all saved optimized prompts."""
    if not PROMPTS_DIR.exists():
        print("No optimized prompts found.")
        return

    for d in sorted(PROMPTS_DIR.iterdir()):
        if d.is_dir() and (d / "metadata.json").exists():
            meta = json.loads((d / "metadata.json").read_text())
            print(f"  {meta['model']} (budget={meta['budget']}, subset={meta['subset']})")


def main():
    parser = argparse.ArgumentParser(description="GEPA prompt optimization per teacher model")
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Student model ID (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--reflection-model",
        default=None,
        help="Stronger model for GEPA reflection (default: same as --model)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="LM API base URL (default: local vLLM)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (not needed for local vLLM, defaults to APIKEY env var for OpenRouter)",
    )
    parser.add_argument("--budget", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--subset", type=int, default=150, help="Total problems for optimization")
    parser.add_argument("--val-size", type=int, default=50, help="Number of validation problems")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument(
        "--max-iters", type=int, default=5, help="Max ReAct tool-calling iterations"
    )
    parser.add_argument("--threads", type=int, default=2, help="Parallel threads for GEPA")
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=3,
        help="Examples per reflection cycle (default 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, recommended for gpt-oss)",
    )
    parser.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help='JSON extra_body for student LM (e.g. \'{"chat_template_kwargs": {"enable_thinking": true}}\')',
    )
    parser.add_argument(
        "--thinking-react",
        action="store_true",
        default=False,
        help="Use ThinkingReAct (native LM thinking) instead of standard ReAct",
    )
    parser.add_argument("--list", action="store_true", help="List saved optimized prompts")
    args = parser.parse_args()

    if args.list:
        list_saved_prompts()
        return

    # Detect if student model is remote (OpenRouter) or local
    is_remote = args.model.startswith("openrouter/")
    if args.api_key:
        api_key = args.api_key
    elif is_remote:
        api_key = os.environ.get("APIKEY", os.environ.get("OPENROUTER_API_KEY", ""))
        if not api_key:
            parser.error("API key required for OpenRouter models (--api-key or APIKEY env var)")
    else:
        api_key = "local"

    run_optimization(
        model_id=args.model,
        api_key=api_key,
        base_url=args.base_url,
        reflection_model=args.reflection_model,
        budget=args.budget,
        subset=args.subset,
        val_size=args.val_size,
        difficulty=args.difficulty,
        max_iters=args.max_iters,
        num_threads=args.threads,
        reflection_minibatch_size=args.reflection_minibatch_size,
        temperature=args.temperature,
        extra_body=args.extra_body,
        use_thinking=args.thinking_react,
    )


if __name__ == "__main__":
    main()
