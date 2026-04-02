"""
Test the SFT model on unseen problems with real tool execution.

Uses the OpenAI Responses API (/v1/responses) which correctly handles
gpt-oss Harmony format tool calls (unlike /v1/chat/completions).

Usage:
    # Test on all unseen problems (not in SFT dataset)
    uv run --no-sync python scripts/test_sft_model.py

    # Test specific problem
    uv run --no-sync python scripts/test_sft_model.py --problem dsp/goertzel

    # Test with /nothink
    uv run --no-sync python scripts/test_sft_model.py --nothink

    # Test N random unseen problems
    uv run --no-sync python scripts/test_sft_model.py --n 10

    # Custom model endpoint
    uv run --no-sync python scripts/test_sft_model.py --base-url http://localhost:8003/v1
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.dataset.loader import discover_pairs
from cnake_charmer.training.environment import CythonToolEnvironment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOOLS_FILE = Path("data/tools.json")
SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")
SFT_DATASET = Path("data/sft_dataset.jsonl")

# Responses API tool format
RESPONSE_TOOLS = [
    {
        "type": "function",
        "name": "evaluate_cython",
        "description": (
            "Compile, analyze, test, and benchmark Cython code in one step. "
            "Returns compilation status, annotation score, correctness tests, and speedup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Complete .pyx source code."},
            },
            "required": ["code"],
        },
    }
]


def parse_metrics(result: str) -> dict:
    """Extract metrics from evaluate_cython result."""
    metrics = {
        "compiled": "Compilation successful" in result,
        "tests_passed": 0,
        "tests_total": 0,
        "annotation": 0.0,
        "speedup": 0.0,
    }
    for line in result.split("\n"):
        m = re.search(r"Tests: (\d+)/(\d+) passed", line)
        if m:
            metrics["tests_passed"] = int(m.group(1))
            metrics["tests_total"] = int(m.group(2))
        m = re.search(r"Annotation score: ([\d.]+)", line)
        if m:
            metrics["annotation"] = float(m.group(1))
        m = re.search(r"Speedup: ([\d.]+)x", line)
        if m:
            metrics["speedup"] = float(m.group(1))
    return metrics


def run_problem(
    base_url,
    model,
    system_prompt,
    problem,
    nothink=False,
    max_iters=5,
    verbose=True,
    reasoning_effort="medium",
):
    """Run ReAct loop using the Responses API (/v1/responses)."""
    env = CythonToolEnvironment()
    env.reset(
        python_code=problem.python_code,
        func_name=problem.func_name,
        test_cases=problem.test_cases,
        benchmark_args=problem.benchmark_args,
    )

    user_content = (
        f"python_code: {problem.python_code}\n\n"
        f"func_name: {problem.func_name}\n"
        f"description: {problem.description or ''}"
    )
    if nothink:
        user_content += " /nothink"

    # Build Responses API request
    request = {
        "model": model,
        "instructions": system_prompt,
        "input": user_content,
        "tools": RESPONSE_TOOLS,
        "max_output_tokens": 8192,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}

    best_metrics = None
    iterations_used = 0
    called_finish = False
    client = httpx.Client(base_url=base_url, timeout=120)

    for iteration in range(max_iters):
        try:
            resp = client.post("/responses", json=request)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            if verbose:
                logger.warning(f"  iter {iteration}: API error: {e}")
            break

        # Process output items
        tool_call = None
        for item in result.get("output", []):
            if item.get("type") == "function_call" and item.get("name") == "evaluate_cython":
                tool_call = item

        if not tool_call:
            if verbose:
                logger.info(f"  iter {iteration}: no tool call (status={result.get('status')})")
            break

        # Execute the tool
        try:
            args = json.loads(tool_call["arguments"])
            code = args.get("code", "")
        except (json.JSONDecodeError, KeyError):
            if verbose:
                logger.warning(f"  iter {iteration}: bad tool args")
            break

        try:
            eval_result = env.safe_evaluate(code=code)
        except Exception as e:
            eval_result = f"## Error\n{e}"

        metrics = parse_metrics(eval_result)
        iterations_used = iteration + 1

        if best_metrics is None or (
            metrics["tests_passed"] > best_metrics["tests_passed"]
            or (
                metrics["tests_passed"] == best_metrics["tests_passed"]
                and metrics["speedup"] > best_metrics["speedup"]
            )
        ):
            best_metrics = metrics

        if verbose:
            status = "OK" if metrics["compiled"] else "FAIL"
            tests = f"{metrics['tests_passed']}/{metrics['tests_total']}"
            logger.info(
                f"  iter {iteration}: {status} | tests {tests} | "
                f"ann {metrics['annotation']:.2f} | {metrics['speedup']:.1f}x"
            )

        # Continue conversation with tool result
        request["input"] = [
            {"type": "message", "role": "user", "content": user_content},
        ]
        # Add previous output items
        for item in result.get("output", []):
            request["input"].append(item)
        # Add tool result
        request["input"].append(
            {
                "type": "function_call_output",
                "call_id": tool_call.get("call_id", ""),
                "output": eval_result,
            }
        )

    client.close()

    return {
        "problem_id": problem.problem_id,
        "difficulty": problem.difficulty,
        "iterations": iterations_used,
        "called_finish": called_finish,
        "compiled": best_metrics["compiled"] if best_metrics else False,
        "tests_passed": best_metrics["tests_passed"] if best_metrics else 0,
        "tests_total": best_metrics["tests_total"] if best_metrics else 0,
        "correct": (
            best_metrics["tests_passed"] == best_metrics["tests_total"] > 0
            if best_metrics
            else False
        ),
        "annotation": best_metrics["annotation"] if best_metrics else 0.0,
        "speedup": best_metrics["speedup"] if best_metrics else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test SFT model on Cython problems")
    parser.add_argument("--base-url", default="http://localhost:8003/v1")
    parser.add_argument("--model", default="gpt-oss-20b-cython")
    parser.add_argument("--problem", default=None, help="Specific problem ID to test")
    parser.add_argument("--n", type=int, default=None, help="Test N random unseen problems")
    parser.add_argument("--nothink", action="store_true", help="Use /nothink mode")
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort level (default: medium)",
    )
    parser.add_argument("--max-iters", type=int, default=5, help="Max ReAct iterations per problem")
    parser.add_argument(
        "--include-seen", action="store_true", help="Include problems from SFT dataset"
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT_FILE.read_text().strip()

    # Load problems
    all_problems = {p.problem_id: p for p in discover_pairs()}

    if args.problem:
        if args.problem not in all_problems:
            logger.error(f"Problem {args.problem} not found")
            return
        test_problems = [all_problems[args.problem]]
    else:
        if not args.include_seen and SFT_DATASET.exists():
            with open(SFT_DATASET) as f:
                sft_pids = set(json.loads(line).get("problem_id") for line in f)
            test_problems = [p for p in all_problems.values() if p.problem_id not in sft_pids]
            logger.info(f"Unseen problems: {len(test_problems)}")
        else:
            test_problems = list(all_problems.values())

        if args.difficulty:
            test_problems = [p for p in test_problems if p.difficulty == args.difficulty]

        if args.n:
            random.seed(args.seed)
            test_problems = random.sample(test_problems, min(args.n, len(test_problems)))

    logger.info(
        f"Testing {len(test_problems)} problems | model={args.model} | "
        f"nothink={args.nothink} | effort={args.effort} | max_iters={args.max_iters}"
    )

    results = []
    for i, problem in enumerate(test_problems):
        logger.info(f"\n[{i + 1}/{len(test_problems)}] {problem.problem_id} ({problem.difficulty})")
        try:
            result = run_problem(
                args.base_url,
                args.model,
                system_prompt,
                problem,
                nothink=args.nothink,
                max_iters=args.max_iters,
                reasoning_effort=args.effort,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"  CRASHED: {e}")
            results.append(
                {
                    "problem_id": problem.problem_id,
                    "difficulty": problem.difficulty,
                    "iterations": 0,
                    "called_finish": False,
                    "compiled": False,
                    "tests_passed": 0,
                    "tests_total": 0,
                    "correct": False,
                    "annotation": 0.0,
                    "speedup": 0.0,
                }
            )

    # Summary
    n = len(results)
    if n == 0:
        logger.info("No results.")
        return

    compiled = sum(1 for r in results if r["compiled"])
    correct = sum(1 for r in results if r["correct"])
    speedups = [r["speedup"] for r in results if r["correct"] and r["speedup"] > 0]
    annotations = [r["annotation"] for r in results if r["correct"]]
    iters_list = [r["iterations"] for r in results]
    avg_iters = sum(iters_list) / n

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {n} problems")
    print(f"{'=' * 60}")
    print(f"  Compiled:    {compiled}/{n} ({compiled / n * 100:.0f}%)")
    print(f"  Correct:     {correct}/{n} ({correct / n * 100:.0f}%)")
    print(f"  Tool calls:  avg={avg_iters:.1f}, min={min(iters_list)}, max={max(iters_list)}")
    if speedups:
        speedups.sort()
        print(
            f"  Speedup (correct): median={speedups[len(speedups) // 2]:.1f}x, "
            f"mean={sum(speedups) / len(speedups):.1f}x, "
            f"max={max(speedups):.1f}x"
        )
    if annotations:
        print(f"  Annotation (correct): mean={sum(annotations) / len(annotations):.2f}")

    # Per-difficulty breakdown
    for diff in ["easy", "medium", "hard"]:
        dr = [r for r in results if r["difficulty"] == diff]
        if not dr:
            continue
        dc = sum(1 for r in dr if r["correct"])
        ds = [r["speedup"] for r in dr if r["correct"] and r["speedup"] > 0]
        di = [r["iterations"] for r in dr]
        line = (
            f"  {diff}: {dc}/{len(dr)} correct, "
            f"calls avg={sum(di) / len(di):.1f} min={min(di)} max={max(di)}"
        )
        if ds:
            line += f", median speedup {sorted(ds)[len(ds) // 2]:.1f}x"
        print(line)

    # Save results
    out = Path("data/sft_test_results.jsonl")
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
