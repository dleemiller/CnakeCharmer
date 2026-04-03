"""
Collect traces from any model/prompt combo into a standardized JSONL format.

Output files can be concatenated, then filtered/combined using dspy-data tools.
Every trace includes model, prompt_id, problem metadata for full traceability.

Usage:
    # 10 random problems with seed prompt on local gpt-oss
    uv run python scripts/collect_traces.py \
        --model openai/gpt-oss-120b \
        --n-random 10

    # Specific problems with a saved program
    uv run python scripts/collect_traces.py \
        --model openrouter/z-ai/glm-5 \
        --problems algorithms/primes,nn_ops/gemm \
        --program data/optimized_prompts/openai_gpt-oss-120b/program.json

    # All problems, 5 attempts each
    uv run python scripts/collect_traces.py \
        --model openai/gpt-oss-120b \
        --all --attempts 5 \
        --output data/traces/gptoss_seed_all.jsonl

    # Failed problems from another run
    uv run python scripts/collect_traces.py \
        --model openrouter/z-ai/glm-5 \
        --problems-from-file failed_problems.txt \
        --attempts 10
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dspy-data-module" / "src"))

from dspy_data.loader import extract_tool_calls

from cnake_charmer.dataset.loader import discover_pairs
from cnake_charmer.training.dspy_agent import CythonOptimization, make_tools
from cnake_charmer.training.rollout import extract_code_from_content

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED_PROMPT = Path(__file__).parent.parent / "data" / "optimized_prompts" / "seed_prompt.txt"

# ---- Standardized trace format ----
# Every JSONL line has these fields. Do not change without versioning.
TRACE_VERSION = "1.0"


def make_trace_record(
    problem,
    result,
    model: str,
    prompt_id: str,
    attempt: int,
    lm_history: list | None = None,
) -> dict:
    """Build a standardized trace record from a DSPy ReAct result."""
    traj = result.trajectory if result.trajectory else {}
    cython_code = extract_code_from_content(getattr(result, "cython_code", "") or "")

    # Fallback: extract code from the last tool call if output field is empty.
    # Some models (e.g. Gemma 4) don't produce a clean final output with the
    # cython_code field, but the code is in the tool call trajectory.
    if not cython_code:
        for i in range(20, -1, -1):
            args = traj.get(f"tool_args_{i}")
            if args and isinstance(args, dict) and "code" in args:
                cython_code = args["code"]
                break
            elif args and isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if "code" in parsed:
                        cython_code = parsed["code"]
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

    # Count iterations (DSPy ReAct may use 0-indexed or 1-indexed keys)
    num_iters = 0
    while f"tool_name_{num_iters}" in traj:
        num_iters += 1
    if num_iters == 0:
        # Try 1-indexed (newer DSPy versions)
        while f"tool_name_{num_iters + 1}" in traj:
            num_iters += 1

    # Extract tool calls
    entry_for_tools = {"trajectory": dict(traj)}
    calls = extract_tool_calls(entry_for_tools)

    # Extract reasoning_content from LM history (thinking models)
    traj = dict(traj)
    if lm_history:
        for idx, interaction in enumerate(lm_history):
            response = interaction.get("response")
            if response is None:
                continue
            for choice in getattr(response, "choices", []):
                rc = getattr(getattr(choice, "message", None), "reasoning_content", None)
                if rc:
                    traj[f"reasoning_{idx}"] = rc

    return {
        "version": TRACE_VERSION,
        "model": model,
        "prompt_id": prompt_id,
        "problem_id": problem.problem_id,
        "func_name": problem.func_name,
        "category": problem.category,
        "difficulty": problem.difficulty,
        "attempt": attempt,
        "num_iterations": num_iters,
        "tools_used": list({c["tool_name"] for c in calls}),
        "trajectory": traj,
        "cython_code": cython_code,
        "output": dict(result) if result else None,
        "reward": None,  # filled in by score_trace
        "timestamp": datetime.now(UTC).isoformat(),
    }


def score_trace(record: dict, problem) -> float:
    """Score the generated cython code using composite reward."""
    from cnake_charmer.training.dspy_agent import _safe_composite_reward

    code = record.get("cython_code", "")
    if not code:
        return 0.0

    scores = _safe_composite_reward(
        cython_code=code,
        python_code=problem.python_code,
        func_name=problem.func_name,
        test_cases=problem.test_cases,
        benchmark_args=problem.benchmark_args,
    )
    return scores["total"]


def load_prompt(program_path: str | None) -> tuple[object | None, str]:
    """Load a GEPA program or seed prompt. Returns (program, prompt_id)."""
    if program_path:
        path = Path(program_path)
        if not path.exists():
            logger.error(f"Program not found: {path}")
            sys.exit(1)
        from scripts.optimize_prompt import CythonReActAgent

        agent = CythonReActAgent(max_iters=5)
        agent.load(path)
        prompt_id = path.stem
        logger.info(f"Loaded program: {path} (prompt_id={prompt_id})")
        return agent, prompt_id

    # Default: seed prompt
    if SEED_PROMPT.exists():
        prompt_id = "seed_v1"
        logger.info(f"Using seed prompt (prompt_id={prompt_id})")
        return None, prompt_id

    prompt_id = "base"
    logger.info("Using base signature (no optimized prompt)")
    return None, prompt_id


def apply_prompt(react_module, optimized_program, seed_text=None):
    """Apply optimized signatures or seed prompt to a ReAct module."""
    if optimized_program is not None:
        opt_params = dict(optimized_program.named_parameters())
        for name, param in react_module.named_parameters():
            if name in opt_params:
                opt = opt_params[name]
                if hasattr(opt, "signature") and hasattr(param, "signature"):
                    param.signature = opt.signature
    elif seed_text:
        for _name, param in react_module.named_parameters():
            if hasattr(param, "signature"):
                param.signature = param.signature.with_instructions(seed_text)


def run_problem(problem, model_id, max_iters, optimized_program, seed_text):
    """Run a single problem and return (result, lm_history)."""
    from copy import deepcopy

    tools, _env = make_tools(
        problem.python_code,
        problem.func_name,
        problem.test_cases,
        problem.benchmark_args,
    )
    react = dspy.ReAct(CythonOptimization, tools=tools, max_iters=max_iters)
    apply_prompt(react, optimized_program, seed_text)

    thread_local_lm = deepcopy(dspy.settings.lm)
    thread_local_lm.history = []
    with dspy.context(lm=thread_local_lm):
        result = react(
            python_code=problem.python_code,
            func_name=problem.func_name,
            description=problem.description or "",
        )
    return result, thread_local_lm.history


def main():
    parser = argparse.ArgumentParser(description="Collect traces into standardized JSONL format")

    # Model
    parser.add_argument(
        "--model", default="openai/gpt-oss-120b", help="Model ID (default: local gpt-oss-120b)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (auto-detected: localhost for local, omit for OpenRouter)",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Prompt
    parser.add_argument(
        "--program", default=None, help="Path to GEPA program.json (default: seed prompt)"
    )
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Override prompt ID label (default: auto from program path)",
    )

    # Problems
    parser.add_argument("--n-random", type=int, default=None, help="Run N random problems")
    parser.add_argument("--problems", default=None, help="Comma-separated problem IDs")
    parser.add_argument(
        "--problems-from-file", default=None, help="File with one problem ID per line"
    )
    parser.add_argument("--all", action="store_true", help="Run all problems")
    parser.add_argument("--shuffle", action="store_true", help="Randomize problem order")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high"],
        help="Set reasoning_effort for thinking models (e.g. Mistral Small 4)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty",
    )

    # Execution
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per problem (default: 1)")
    parser.add_argument(
        "--max-iters", type=int, default=5, help="Max evaluate_cython calls per attempt"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Run N attempts concurrently via dspy.Parallel (use with vLLM prefix caching)",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSONL path (default: data/traces/{model}_{prompt}.jsonl)",
    )

    args = parser.parse_args()

    # Resolve model and API
    is_remote = args.model.startswith("openrouter/")
    if args.api_key:
        api_key = args.api_key
    elif is_remote:
        api_key = os.environ.get("APIKEY", os.environ.get("OPENROUTER_API_KEY", ""))
        if not api_key:
            parser.error("API key required for OpenRouter (set APIKEY env var)")
    else:
        api_key = "local"

    lm_kwargs = {
        "api_key": api_key,
        "temperature": args.temperature,
        "cache": False,
        "max_tokens": 8192,
    }
    if args.reasoning_effort:
        if is_remote:
            # OpenRouter doesn't support reasoning_effort natively; pass via extra_body
            lm_kwargs["extra_body"] = {"reasoning_effort": args.reasoning_effort}
        else:
            lm_kwargs["reasoning_effort"] = args.reasoning_effort
    if not is_remote:
        lm_kwargs["api_base"] = args.base_url or "http://localhost:8000/v1"

    lm = dspy.LM(args.model, **lm_kwargs)
    dspy.settings.configure(lm=lm)
    logger.info(f"Model: {args.model}")

    # Load prompt
    optimized_program, prompt_id = load_prompt(args.program)
    if args.prompt_id:
        prompt_id = args.prompt_id
    seed_text = (
        SEED_PROMPT.read_text().strip() if SEED_PROMPT.exists() and not optimized_program else None
    )

    # Resolve problems
    all_problems = {p.problem_id: p for p in discover_pairs()}
    if args.difficulty:
        all_problems = {k: v for k, v in all_problems.items() if v.difficulty == args.difficulty}

    if args.problems:
        problem_ids = [p.strip() for p in args.problems.split(",")]
        problems = [all_problems[pid] for pid in problem_ids if pid in all_problems]
    elif args.problems_from_file:
        with open(args.problems_from_file) as f:
            problem_ids = [line.strip() for line in f if line.strip()]
        problems = [all_problems[pid] for pid in problem_ids if pid in all_problems]
    elif args.all:
        problems = list(all_problems.values())
    elif args.n_random:
        problems = random.sample(list(all_problems.values()), min(args.n_random, len(all_problems)))
    else:
        problems = random.sample(list(all_problems.values()), min(10, len(all_problems)))

    logger.info(f"Problems: {len(problems)}, Attempts: {args.attempts}, Prompt: {prompt_id}")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_slug = args.model.replace("/", "_").replace(":", "_")
        output_path = Path(f"data/traces/{model_slug}_{prompt_id}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count existing traces per problem for this model
    # Normalize model names: strip :free suffix so paid/free variants match
    def normalize_model(m: str) -> str:
        return m.removesuffix(":free")

    existing_counts = Counter()
    model_norm = normalize_model(args.model)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if normalize_model(r.get("model", "")) == model_norm:
                            existing_counts[r.get("problem_id", "")] += 1
                    except json.JSONDecodeError:
                        pass
        if existing_counts:
            logger.info(
                f"Resuming: {sum(existing_counts.values())} existing traces for {args.model} across {len(existing_counts)} problems"
            )

    # Build work list, skipping completed problems
    work = []
    skipped = 0
    for problem in problems:
        existing = existing_counts.get(problem.problem_id, 0)
        remaining = args.attempts - existing
        if remaining <= 0:
            skipped += 1
            continue
        for attempt in range(existing, existing + remaining):
            work.append((problem, attempt))

    if skipped:
        logger.info(f"Skipping {skipped} complete problems, {len(work)} traces remaining")

    if args.shuffle:
        random.shuffle(work)

    # Run
    total = len(work)
    done = 0

    if args.parallel:
        # Parallel execution using dspy.Parallel — ideal for vLLM prefix caching.
        # Groups attempts by problem so same-prefix requests hit the KV cache.
        from itertools import groupby

        logger.info(f"Running {total} traces with {args.parallel} parallel threads")

        # Group work by problem for prefix cache locality
        work_sorted = sorted(work, key=lambda x: x[0].problem_id)

        for pid, group in groupby(work_sorted, key=lambda x: x[0].problem_id):
            group_items = list(group)
            problem = group_items[0][0]

            # Build exec pairs: each attempt gets its own module + example
            exec_pairs = []
            for prob, attempt in group_items:
                module = dspy.Predict("question -> answer")  # placeholder, run_problem handles it
                example = dspy.Example(
                    problem=prob,
                    attempt=attempt,
                    model_id=args.model,
                    max_iters=args.max_iters,
                    optimized_program=optimized_program,
                    seed_text=seed_text,
                ).with_inputs(
                    "problem", "attempt", "model_id", "max_iters", "optimized_program", "seed_text"
                )
                exec_pairs.append((module, example))

            # Define a wrapper that dspy.Parallel can call
            class TraceRunner:
                def __call__(
                    self,
                    problem,
                    attempt,
                    model_id,
                    max_iters,
                    optimized_program,
                    seed_text,
                    **kwargs,
                ):
                    result, lm_history = run_problem(
                        problem, model_id, max_iters, optimized_program, seed_text
                    )
                    return dspy.Example(
                        result=result, lm_history=lm_history, problem=problem, attempt=attempt
                    )

            runner = TraceRunner()
            exec_pairs = [(runner, ex) for _, ex in exec_pairs]

            parallel = dspy.Parallel(
                num_threads=min(args.parallel, len(exec_pairs)),
                max_errors=len(exec_pairs),  # don't stop on errors
                disable_progress_bar=False,
                timeout=300,
            )
            try:
                results = parallel(exec_pairs=exec_pairs)
            except Exception as e:
                logger.error(f"  Parallel failed for {pid}: {e}")
                results = []

            for res in results:
                if res is None:
                    continue
                try:
                    record = make_trace_record(
                        res.problem, res.result, args.model, prompt_id, res.attempt, res.lm_history
                    )
                    record["reward"] = score_trace(record, res.problem)
                    with open(output_path, "a") as f:
                        f.write(json.dumps(record, default=str) + "\n")
                    done += 1
                    logger.info(
                        f"  [{done}/{total}] {res.problem.problem_id} attempt {res.attempt + 1}: "
                        f"reward={record['reward']:.3f} iters={record['num_iterations']}"
                    )
                except Exception as e:
                    logger.error(f"  Record failed: {e}")
    else:
        # Sequential execution
        for problem, attempt in work:
            done += 1
            logger.info(
                f"[{done}/{total}] {problem.problem_id} attempt {attempt + 1}/{args.attempts}"
            )
            try:
                result, lm_history = run_problem(
                    problem, args.model, args.max_iters, optimized_program, seed_text
                )
                record = make_trace_record(
                    problem, result, args.model, prompt_id, attempt, lm_history
                )
                record["reward"] = score_trace(record, problem)

                # Append
                with open(output_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")

                logger.info(
                    f"  reward={record['reward']:.3f} iters={record['num_iterations']} "
                    f"tools={record['tools_used']}"
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")

    # Summary
    logger.info(f"\nSaved {done} traces to {output_path}")


if __name__ == "__main__":
    main()
