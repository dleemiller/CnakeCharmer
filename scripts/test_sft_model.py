"""
Test the SFT model on unseen problems with real tool execution.

Runs a ReAct loop against the model via OpenAI API, calling evaluate_cython
for real compilation/testing/benchmarking.

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
    uv run --no-sync python scripts/test_sft_model.py --base-url http://localhost:8003/v1 --model gpt-oss-20b-cython
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.dataset.loader import discover_pairs
from cnake_charmer.training.environment import CythonToolEnvironment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOOLS_FILE = Path("data/tools.json")
SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")
SFT_DATASET = Path("data/sft_dataset.jsonl")


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
    client,
    model,
    system_prompt,
    tools,
    problem,
    nothink=False,
    max_iters=5,
    verbose=True,
    reasoning_effort="medium",
):
    """Run ReAct loop on a single problem. Returns dict with results."""
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    best_metrics = None
    iterations_used = 0
    called_finish = False

    for iteration in range(max_iters):
        try:
            extra_body = {}
            if reasoning_effort:
                extra_body["chat_template_kwargs"] = {"effort": reasoning_effort}
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=4096,
                temperature=0.7,
                timeout=60,
                extra_body=extra_body if extra_body else None,
            )
        except Exception as e:
            if verbose:
                logger.warning(f"  iter {iteration}: API error: {e}")
            break

        msg = response.choices[0].message

        if not msg.tool_calls:
            if verbose:
                logger.info(f"  iter {iteration}: stop ({response.choices[0].finish_reason})")
            break

        tc = msg.tool_calls[0]
        fn = tc.function
        args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments

        if "finish" in fn.name:
            called_finish = True
            if verbose:
                logger.info(f"  iter {iteration}: finish()")
            break

        code = args.get("code", "")
        try:
            result = env.safe_evaluate(code=code)
        except Exception as e:
            result = f"## Error\n{e}"

        metrics = parse_metrics(result)
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

        messages.append(msg.model_dump())
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

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
    parser.add_argument("--api-key", default="empty")
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

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    system_prompt = SYSTEM_PROMPT_FILE.read_text().strip()
    tools = json.loads(TOOLS_FILE.read_text())

    # Load problems
    all_problems = {p.problem_id: p for p in discover_pairs()}

    if args.problem:
        if args.problem not in all_problems:
            logger.error(f"Problem {args.problem} not found")
            return
        test_problems = [all_problems[args.problem]]
    else:
        # Filter to unseen by default
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
        f"nothink={args.nothink} | max_iters={args.max_iters}"
    )

    results = []
    for i, problem in enumerate(test_problems):
        logger.info(f"\n[{i + 1}/{len(test_problems)}] {problem.problem_id} ({problem.difficulty})")
        try:
            result = run_problem(
                client,
                args.model,
                system_prompt,
                tools,
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
    finished = sum(1 for r in results if r["called_finish"])
    speedups = [r["speedup"] for r in results if r["correct"] and r["speedup"] > 0]
    annotations = [r["annotation"] for r in results if r["correct"]]
    iters_list = [r["iterations"] for r in results]
    avg_iters = sum(iters_list) / n

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {n} problems")
    print(f"{'=' * 60}")
    print(f"  Compiled:    {compiled}/{n} ({compiled / n * 100:.0f}%)")
    print(f"  Correct:     {correct}/{n} ({correct / n * 100:.0f}%)")
    print(f"  Called finish: {finished}/{n} ({finished / n * 100:.0f}%)")
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
        line = f"  {diff}: {dc}/{len(dr)} correct, calls avg={sum(di) / len(di):.1f} min={min(di)} max={max(di)}"
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
