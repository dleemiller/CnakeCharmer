"""
Generate SFT training traces using GEPA-optimized DSPy ReAct agent.

Generates N trace candidates per problem, scores each with composite
reward, and keeps the best one per problem for SFT training.

Usage:
    # Local vLLM with optimized prompt (default)
    uv run python scripts/generate_traces.py --n 5 --subset 50

    # Full dataset
    uv run python scripts/generate_traces.py --n 5

    # OpenRouter
    APIKEY=... uv run python scripts/generate_traces.py \
        --model openrouter/deepseek/deepseek-v3.2 \
        --base-url https://openrouter.ai/api/v1 --n 5
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.training.dspy_agent import (
    CythonOptimization,
    collect_reward,
    make_tools,
)
from cnake_data.loader import discover_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "optimized_prompts"


class PerProblemReAct(dspy.Module):
    """Wrapper that creates fresh tools per problem and applies optimized signatures."""

    def __init__(self, max_iters: int = 5, optimized_program=None):
        super().__init__()
        self.max_iters = max_iters
        self.optimized_program = optimized_program

    def forward(
        self,
        python_code: str,
        func_name: str,
        description: str = "",
        test_cases: str = "[]",
        benchmark_args: str = "null",
    ):
        tc = json.loads(test_cases) if isinstance(test_cases, str) else test_cases
        ba = json.loads(benchmark_args) if isinstance(benchmark_args, str) else benchmark_args

        tools, _env = make_tools(python_code, func_name, tc, ba)
        react = dspy.ReAct(CythonOptimization, tools=tools, max_iters=self.max_iters)

        # Apply optimized signatures from GEPA if available
        if self.optimized_program is not None:
            _apply_optimized_signatures(react, self.optimized_program)

        return react(python_code=python_code, func_name=func_name, description=description)


def _apply_optimized_signatures(react_module, optimized_program):
    """Copy optimized signatures from a saved GEPA program to a fresh ReAct module."""
    opt_params = dict(optimized_program.named_parameters())
    for name, param in react_module.named_parameters():
        if name in opt_params:
            opt_param = opt_params[name]
            if hasattr(opt_param, "signature") and hasattr(param, "signature"):
                param.signature = opt_param.signature


def load_optimized_program(model_id: str) -> dspy.Module | None:
    """Load a GEPA-optimized program for a model, if available."""
    slug = model_id.replace("/", "_").replace(":", "_")
    program_path = PROMPTS_DIR / slug / "program.json"

    if not program_path.exists():
        logger.info(f"No optimized program found at {program_path}")
        return None

    # Load into a placeholder CythonReActAgent to get the signatures
    from scripts.optimize_prompt import CythonReActAgent

    agent = CythonReActAgent(max_iters=5)
    agent.load(program_path)
    logger.info(f"Loaded optimized program from {program_path}")
    return agent


def select_best_traces(entries: list[dict], require_all_tools: bool = False) -> list[dict]:
    """From N traces per problem, keep only the best-scoring one per problem.

    When require_all_tools=True, prefers traces that used all 4 tools.
    Falls back to best reward if no trace used all tools.
    """
    from cnake_charmer.training.dspy_agent import REQUIRED_TOOLS, extract_tool_usage

    by_problem = defaultdict(list)
    for entry in entries:
        key = entry.get("inputs", {}).get("func_name", "unknown")
        by_problem[key].append(entry)

    best = []
    full_tool_count = 0
    for problem_key, candidates in sorted(by_problem.items()):
        if require_all_tools:
            # Prefer traces that used all 4 tools
            full_tool = [
                e for e in candidates if extract_tool_usage(e)["tools_used"] >= REQUIRED_TOOLS
            ]
            if full_tool:
                scored = [(e.get("reward") or 0.0, e) for e in full_tool]
                full_tool_count += 1
            else:
                scored = [(e.get("reward") or 0.0, e) for e in candidates]
        else:
            scored = [(e.get("reward") or 0.0, e) for e in candidates]

        scored.sort(key=lambda x: x[0], reverse=True)
        best_reward, best_entry = scored[0]
        best.append(best_entry)
        n = len(candidates)
        rewards = [f"{s[0]:.3f}" for s in scored]
        logger.info(
            f"  {problem_key}: best {best_reward:.3f} from {n} candidates ({', '.join(rewards)})"
        )

    if require_all_tools:
        logger.info(f"  {full_tool_count}/{len(by_problem)} problems had traces with all 4 tools")

    return best


def run_collection(
    problems,
    output_path: str,
    n: int = 5,
    num_threads: int = 8,
    max_iters: int = 5,
    optimized_program=None,
):
    """Generate N traces per problem using dspy-data's Collect.

    Supports resume: checks existing JSONL for completed problem/run pairs
    and skips them. Safe to restart after crashes.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dspy-data-module" / "src"))
    from dspy_data import Collect
    from dspy_data.loader import load_collected

    # Check existing traces for resume
    existing = load_collected(output_path) if Path(output_path).exists() else []
    existing_counts = defaultdict(int)
    for entry in existing:
        key = entry.get("inputs", {}).get("func_name", "")
        existing_counts[key] += 1

    # Build examples, skipping already-completed ones
    examples = []
    skipped = 0
    for p in problems:
        already = existing_counts.get(p.func_name, 0)
        remaining = n - already
        if remaining <= 0:
            skipped += 1
            continue
        for _ in range(remaining):
            examples.append(
                {
                    "python_code": p.python_code,
                    "func_name": p.func_name,
                    "description": p.description or "",
                    "test_cases": json.dumps(p.test_cases),
                    "benchmark_args": json.dumps(p.benchmark_args),
                }
            )

    if skipped:
        logger.info(
            f"Resuming: {skipped} problems already complete, {len(examples)} traces remaining"
        )

    if not examples:
        logger.info("All traces already generated!")
        return []

    predictor = PerProblemReAct(max_iters=max_iters, optimized_program=optimized_program)

    collector = Collect(
        predictor=predictor,
        output_dir=output_path,
        reward_fn=collect_reward,
        num_threads=num_threads,
        output_format="jsonl",
    )

    logger.info(f"Generating {len(examples)} traces ({n} per problem, {len(problems)} problems)...")
    results = collector(examples, n=1)  # n=1 since we already expanded
    logger.info(f"Collection complete. {len(results)} successful traces.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training traces")
    parser.add_argument(
        "--model", default="openai/gpt-oss-120b", help="Model ID (default: openai/gpt-oss-120b)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="LM API base URL (default: local vLLM)",
    )
    parser.add_argument("--api-key", default=None, help="API key (not needed for local vLLM)")
    parser.add_argument(
        "--output",
        default="data/trajectories/traces.jsonl",
        help="Output JSONL path (all candidates)",
    )
    parser.add_argument(
        "--best-output",
        default="data/trajectories/best_traces.jsonl",
        help="Output JSONL path (best per problem)",
    )
    parser.add_argument("--n", type=int, default=5, help="Trace candidates per problem")
    parser.add_argument("--subset", type=int, default=None, help="Use N problems (for prototyping)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument(
        "--max-iters", type=int, default=5, help="Max ReAct tool-calling iterations"
    )
    parser.add_argument("--threads", type=int, default=8, help="Parallel threads")
    parser.add_argument(
        "--no-optimized-prompt", action="store_true", help="Skip loading GEPA-optimized prompt"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, recommended for gpt-oss)",
    )
    args = parser.parse_args()

    # Configure DSPy LM
    if args.api_key:
        api_key = args.api_key
    elif "localhost" in args.base_url or "127.0.0.1" in args.base_url:
        api_key = "local"
    else:
        api_key = os.environ.get("APIKEY", os.environ.get("OPENROUTER_API_KEY", "local"))

    lm = dspy.LM(
        args.model,
        api_base=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        cache=False,  # disable DSPy cache so N traces per problem are diverse
    )
    dspy.settings.configure(lm=lm)
    logger.info(f"Configured LM: {args.model} @ {args.base_url}")

    # Load GEPA-optimized program if available
    optimized = None
    if not args.no_optimized_prompt:
        optimized = load_optimized_program(args.model)

    # Load problems
    problems = discover_pairs()
    logger.info(f"Discovered {len(problems)} problems")

    if args.difficulty:
        problems = [p for p in problems if p.difficulty == args.difficulty]
        logger.info(f"Filtered to {len(problems)} {args.difficulty} problems")

    if args.subset:
        problems = problems[: args.subset]
        logger.info(f"Using subset of {len(problems)} problems")

    if not problems:
        logger.error("No problems found!")
        return

    # Generate traces
    run_collection(
        problems,
        output_path=args.output,
        n=args.n,
        num_threads=args.threads,
        max_iters=args.max_iters,
        optimized_program=optimized,
    )

    # Load all traces and select best per problem
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dspy-data-module" / "src"))
    from dspy_data import collected_stats, load_collected

    entries = load_collected(args.output)
    logger.info("\nAll traces stats:")
    stats = collected_stats(entries)
    logger.info(json.dumps(stats, indent=2))

    # Select best per problem
    logger.info(f"\nSelecting best of {args.n} per problem...")
    best = select_best_traces(entries)

    # Save best traces
    best_path = Path(args.best_output)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, "w") as f:
        for entry in best:
            f.write(json.dumps(entry, default=str) + "\n")

    best_stats = collected_stats(best)
    logger.info(f"\nBest traces stats ({len(best)} problems):")
    logger.info(json.dumps(best_stats, indent=2))

    # Summary
    all_rewards = [e.get("reward", 0) or 0 for e in entries]
    best_rewards = [e.get("reward", 0) or 0 for e in best]
    logger.info(f"\nReward improvement from best-of-{args.n} selection:")
    logger.info(f"  All traces mean: {sum(all_rewards) / len(all_rewards):.3f}")
    logger.info(f"  Best traces mean: {sum(best_rewards) / len(best_rewards):.3f}")


if __name__ == "__main__":
    main()
