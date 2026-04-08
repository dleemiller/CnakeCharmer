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
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.traces.io import append_trace
from cnake_charmer.traces.lm import (
    apply_optimized_signatures,
    configure_dspy_lm,
    get_seed_text,
    load_optimized_prompt,
    model_slug,
)
from cnake_charmer.traces.models import ToolStep, Trace
from cnake_charmer.training.dspy_agent import CythonOptimization, make_tools
from cnake_charmer.training.rollout import extract_code_from_content
from cnake_data.loader import discover_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_tool_args_raw(args) -> dict:
    """Normalize tool args from DSPy trajectory to dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"raw": args}
    return {}


def _tool_call_counts(trace: Trace) -> str:
    """Format per-tool call counts for logging, e.g. 'evaluate_cython=3 wiki_read=1'."""
    counts = Counter(s.tool_name for s in trace.steps)
    return " ".join(f"{name}={n}" for name, n in counts.most_common())


def make_trace(
    problem,
    result,
    model: str,
    prompt_id: str,
    attempt: int,
    lm_history: list | None = None,
) -> Trace:
    """Build a v2 Trace object from a DSPy ReAct result."""
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

    # Build reasoning map from LM history (thinking models)
    reasoning_map = {}  # idx -> reasoning_content
    if lm_history:
        for idx, interaction in enumerate(lm_history):
            response = interaction.get("response")
            if response is None:
                continue
            for choice in getattr(response, "choices", []):
                rc = getattr(getattr(choice, "message", None), "reasoning_content", None)
                if rc:
                    reasoning_map[idx] = rc

    # Parse tool steps
    steps = []
    i = 0
    while f"tool_name_{i}" in traj:
        tool_name = traj[f"tool_name_{i}"]
        if tool_name is not None:
            steps.append(
                ToolStep(
                    tool_name=str(tool_name),
                    tool_args=_parse_tool_args_raw(traj.get(f"tool_args_{i}", {})),
                    observation=str(traj.get(f"observation_{i}", "")),
                    thought=traj.get(f"thought_{i}"),
                    reasoning=reasoning_map.get(i),
                )
            )
        i += 1

    # Collect trailing reasoning entries (beyond tool steps)
    n_steps = len(steps)
    trailing = []
    j = n_steps
    while j in reasoning_map:
        trailing.append(reasoning_map[j])
        j += 1

    thinking = bool(reasoning_map)

    return Trace(
        problem_id=problem.problem_id,
        model=model,
        prompt_id=prompt_id,
        attempt=attempt,
        timestamp=datetime.now(UTC),
        steps=steps,
        trailing_reasoning=trailing,
        final_code=cython_code,
        reward=0.0,
        thinking=thinking,
        func_name=problem.func_name,
        category=problem.category,
        difficulty=problem.difficulty,
    )


def score_trace(trace: Trace, problem) -> float:
    """Score the generated cython code using composite reward.

    Also populates trace.metrics with detailed scoring breakdown.
    """
    from cnake_charmer.training.dspy_agent import _safe_composite_reward

    code = trace.final_code or ""
    if not code:
        return 0.0

    scores = _safe_composite_reward(
        cython_code=code,
        python_code=problem.python_code,
        func_name=problem.func_name,
        test_cases=problem.test_cases,
        benchmark_args=problem.benchmark_args,
    )
    trace.metrics = {
        "compiled": scores.get("compiled", False),
        "correctness": round(scores.get("correctness", 0.0), 3),
        "speedup": round(scores.get("speedup", 0.0), 2),
        "annotations": round(scores.get("annotations", 0.0), 3),
    }
    return scores["total"]


def _format_trace_log(trace: Trace) -> str:
    """Format a single trace result for logging."""
    m = trace.metrics
    parts = [f"reward={trace.reward:.3f}"]

    if m.get("compiled"):
        parts.append(f"correct={m.get('correctness', 0):.0%}")
        parts.append(f"speedup={m.get('speedup', 0):.1f}x")
        parts.append(f"ann={m.get('annotations', 0):.3f}")
    else:
        parts.append("COMPILE_FAIL")

    parts.append(f"iters={trace.num_iterations}")
    parts.append(f"tools=[{_tool_call_counts(trace)}]")
    return " ".join(parts)


def run_problem(
    problem,
    model_id,
    max_iters,
    optimized_program,
    seed_text,
    use_thinking=False,
    include_wiki=False,
):
    """Run a single problem and return (result, lm_history)."""
    from copy import deepcopy

    tools, _env = make_tools(
        problem.python_code,
        problem.func_name,
        problem.test_cases,
        problem.benchmark_args,
        include_wiki=include_wiki,
    )
    if use_thinking:
        from cnake_charmer.traces.thinking_react import ThinkingReAct

        react = ThinkingReAct(CythonOptimization, tools=tools, max_iters=max_iters)
    else:
        react = dspy.ReAct(CythonOptimization, tools=tools, max_iters=max_iters)
    apply_optimized_signatures(react, optimized_program, seed_text)

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
    parser.add_argument("--top-p", type=float, default=None)

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

    parser.add_argument(
        "--thinking-react",
        action="store_true",
        default=False,
        help="Use ThinkingReAct (native LM thinking) instead of standard ReAct",
    )
    parser.add_argument(
        "--enable-wiki",
        action="store_true",
        default=False,
        help="Add wiki_read and wiki_search tools (capped at 2 calls per problem)",
    )
    parser.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help='JSON extra_body for LM (e.g. \'{"chat_template_kwargs": {"enable_thinking": true}}\')',
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSONL path (default: data/traces/{model}_{prompt}.jsonl)",
    )

    args = parser.parse_args()

    # Configure LM (shared utility handles remote vs local detection)
    lm_extra = {}
    if args.extra_body:
        lm_extra["extra_body"] = args.extra_body
    configure_dspy_lm(
        args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
        **lm_extra,
    )
    logger.info(f"Model: {args.model}")

    # Load prompt
    optimized_program, prompt_id = load_optimized_prompt(
        model_id=args.model, program_path=args.program
    )
    if args.prompt_id:
        prompt_id = args.prompt_id
    seed_text = get_seed_text() if not optimized_program else None

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
        slug = model_slug(args.model)
        output_path = Path(f"data/traces/{slug}_{prompt_id}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count existing traces per problem for this model
    # Normalize model names: strip :free suffix and -preview so variants match
    def normalize_model(m: str) -> str:
        return m.removesuffix(":free").removesuffix("-preview")

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
        logger.info(f"Running {total} traces with {args.parallel} parallel threads")

        # Group work by problem — shuffle already controls problem order
        work_by_pid = {}
        pid_order = []
        for problem, attempt in work:
            pid = problem.problem_id
            if pid not in work_by_pid:
                work_by_pid[pid] = []
                pid_order.append(pid)
            work_by_pid[pid].append((problem, attempt))

        for pid in pid_order:
            group_items = work_by_pid[pid]
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
                    use_thinking=args.thinking_react,
                    include_wiki=args.enable_wiki,
                ).with_inputs(
                    "problem",
                    "attempt",
                    "model_id",
                    "max_iters",
                    "optimized_program",
                    "seed_text",
                    "use_thinking",
                    "include_wiki",
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
                    use_thinking=False,
                    include_wiki=False,
                    **kwargs,
                ):
                    result, lm_history = run_problem(
                        problem,
                        model_id,
                        max_iters,
                        optimized_program,
                        seed_text,
                        use_thinking=use_thinking,
                        include_wiki=include_wiki,
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
                    trace = make_trace(
                        res.problem, res.result, args.model, prompt_id, res.attempt, res.lm_history
                    )
                    trace.reward = score_trace(trace, res.problem)
                    append_trace(trace, output_path)
                    done += 1
                    logger.info(
                        f"  [{done}/{total}] {res.problem.problem_id} "
                        f"attempt {res.attempt + 1}: {_format_trace_log(trace)}"
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
                    problem,
                    args.model,
                    args.max_iters,
                    optimized_program,
                    seed_text,
                    use_thinking=args.thinking_react,
                    include_wiki=args.enable_wiki,
                )
                trace = make_trace(problem, result, args.model, prompt_id, attempt, lm_history)
                trace.reward = score_trace(trace, problem)
                append_trace(trace, output_path)

                logger.info(f"  {_format_trace_log(trace)}")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    # Summary
    logger.info(f"\nSaved {done} traces to {output_path}")


if __name__ == "__main__":
    main()
