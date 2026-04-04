"""
Run a few sample problems through models on OpenRouter using the GEPA-optimized prompt.

Generates traces for manual inspection of agentic coding quality.

Usage:
    # Run 3 problems through a model
    uv run python scripts/sample_openrouter.py \
        --model openrouter/deepseek/deepseek-v3.2 \
        --n-problems 3

    # Run specific problems
    uv run python scripts/sample_openrouter.py \
        --model openrouter/xiaomi/mimo-v2-pro \
        --problems algorithms/primes,numerical/great_circle,nn_ops/gemm

    # List available models
    uv run python scripts/sample_openrouter.py --list-models
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dspy-data-module" / "src"))


from cnake_charmer.training.dspy_agent import (
    CythonOptimization,
    make_tools,
)
from cnake_data.loader import discover_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "optimized_prompts"

# Sample problems spanning easy/medium/hard
DEFAULT_PROBLEMS = [
    "algorithms/primes",  # easy: simple trial division
    "algorithms/bloom_filter",  # medium: bit array + hashing
    "nn_ops/gemm",  # hard: matrix multiply with malloc
]

OPENROUTER_MODELS = {
    "minimax-m2.7": "openrouter/minimax/minimax-m2.7",
    "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
    "kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
    "glm-5": "openrouter/z-ai/glm-5",
    "mimo-v2-pro": "openrouter/xiaomi/mimo-v2-pro",
    "gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
}


def load_optimized_program():
    """Load the GEPA-optimized program if available."""
    program_path = PROMPTS_DIR / "openai_gpt-oss-120b" / "program.json"
    if not program_path.exists():
        logger.warning(f"No optimized program at {program_path}, using default prompt")
        return None

    from scripts.optimize_prompt import CythonReActAgent

    agent = CythonReActAgent(max_iters=5)
    agent.load(program_path)
    logger.info(f"Loaded optimized program from {program_path}")
    return agent


def _apply_optimized_signatures(react_module, optimized_program):
    """Copy optimized signatures from GEPA program to fresh ReAct."""
    if optimized_program is None:
        return
    opt_params = dict(optimized_program.named_parameters())
    for name, param in react_module.named_parameters():
        if name in opt_params:
            opt_param = opt_params[name]
            if hasattr(opt_param, "signature") and hasattr(param, "signature"):
                param.signature = opt_param.signature


def run_problem(problem, max_iters=5, optimized_program=None):
    """Run a single problem and return the trajectory + result."""
    tools, env = make_tools(
        problem.python_code,
        problem.func_name,
        problem.test_cases,
        problem.benchmark_args,
    )
    react = dspy.ReAct(CythonOptimization, tools=tools, max_iters=max_iters)
    _apply_optimized_signatures(react, optimized_program)

    result = react(
        python_code=problem.python_code,
        func_name=problem.func_name,
        description=problem.description or "",
    )
    return result


def display_result(problem, result):
    """Pretty-print a single problem's trace for manual inspection."""
    traj = result.trajectory
    cython_code = getattr(result, "cython_code", "")

    print(f"\n{'=' * 80}")
    print(f"PROBLEM: {problem.problem_id} ({problem.difficulty})")
    print(f"{'=' * 80}")
    print(f"Function: {problem.func_name}")
    print(f"Description: {(problem.description or '')[:100]}")
    print()

    # Show each tool call
    idx = 0
    while f"tool_name_{idx}" in traj:
        thought = traj.get(f"thought_{idx}", "")
        tool = traj[f"tool_name_{idx}"]
        observation = traj.get(f"observation_{idx}", "")

        print(f"--- Step {idx + 1}: {tool} ---")
        if thought:
            print(f"Thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")
        print("Result:")
        for line in observation.split("\n")[:15]:
            print(f"  {line}")
        if observation.count("\n") > 15:
            print(f"  ... ({observation.count(chr(10)) - 15} more lines)")
        print()
        idx += 1

    # Show final code snippet
    if cython_code:
        lines = cython_code.split("\n")
        print(f"--- Final Cython ({len(lines)} lines) ---")
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")
    else:
        print("--- No Cython code produced ---")

    print()


def main():
    parser = argparse.ArgumentParser(description="Sample problems through OpenRouter models")
    parser.add_argument("--model", help="Model shortname or full OpenRouter path")
    parser.add_argument("--n-problems", type=int, default=3, help="Number of problems to sample")
    parser.add_argument(
        "--problems",
        default=None,
        help="Comma-separated problem IDs (e.g., algorithms/primes,nn_ops/gemm)",
    )
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-optimized-prompt", action="store_true")
    parser.add_argument("--list-models", action="store_true", help="List known OpenRouter models")
    parser.add_argument("--output", default=None, help="Save traces to JSONL file")
    args = parser.parse_args()

    if args.list_models:
        print("Known OpenRouter models:")
        for name, path in sorted(OPENROUTER_MODELS.items()):
            print(f"  {name:20s} → {path}")
        return

    if not args.model:
        parser.error("--model required")

    # Resolve model name
    model_id = OPENROUTER_MODELS.get(args.model, args.model)

    # Configure LM
    api_key = os.environ.get("APIKEY", os.environ.get("OPENROUTER_API_KEY", ""))
    if not api_key:
        parser.error("Set APIKEY or OPENROUTER_API_KEY env var")

    lm = dspy.LM(model_id, api_key=api_key, temperature=args.temperature, cache=False)
    dspy.settings.configure(lm=lm)
    logger.info(f"Model: {model_id}")

    # Load problems
    all_problems = {p.problem_id: p for p in discover_pairs()}

    if args.problems:
        problem_ids = [p.strip() for p in args.problems.split(",")]
        problems = [all_problems[pid] for pid in problem_ids if pid in all_problems]
        missing = [pid for pid in problem_ids if pid not in all_problems]
        if missing:
            logger.warning(f"Problems not found: {missing}")
    else:
        # Pick a spread of easy/medium/hard
        by_diff = {"easy": [], "medium": [], "hard": []}
        for p in all_problems.values():
            by_diff.get(p.difficulty, by_diff["medium"]).append(p)
        problems = []
        per = max(1, args.n_problems // 3)
        for diff in ["easy", "medium", "hard"]:
            problems.extend(by_diff[diff][:per])
        problems = problems[: args.n_problems]

    logger.info(f"Running {len(problems)} problems: {[p.problem_id for p in problems]}")

    # Load optimized prompt
    optimized = None if args.no_optimized_prompt else load_optimized_program()

    # Run each problem
    results = []
    for problem in problems:
        logger.info(f"Running {problem.problem_id}...")
        try:
            result = run_problem(problem, max_iters=args.max_iters, optimized_program=optimized)
            display_result(problem, result)
            results.append(
                {
                    "model": model_id,
                    "problem_id": problem.problem_id,
                    "difficulty": problem.difficulty,
                    "trajectory": dict(result.trajectory) if result.trajectory else {},
                    "cython_code": getattr(result, "cython_code", ""),
                }
            )
        except Exception as e:
            logger.error(f"Failed on {problem.problem_id}: {e}")
            import traceback

            traceback.print_exc()

    # Save if requested
    if args.output and results:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        logger.info(f"Saved {len(results)} traces to {args.output}")


if __name__ == "__main__":
    main()
