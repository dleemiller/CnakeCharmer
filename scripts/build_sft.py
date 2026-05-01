"""
Build SFT dataset from collected trace files.

Pipeline: load → score/filter → validate → render + token screen →
          select top-k → effort terciles → validate rendered → save

Usage:
    uv run --no-sync python scripts/build_sft.py --require-finish
    uv run --no-sync python scripts/build_sft.py --min-score 0.5 --top-k 4
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.traces.io import load_traces
from cnake_charmer.traces.models import Trace
from cnake_charmer.training.sft_scoring import parse_trace_metrics, score_trace
from cnake_charmer.training.sft_validation import (
    total_analysis_length,
    validate_rendered_example,
    validate_trace_for_rendering,
)
from cnake_data.loader import discover_pairs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACES_DIR = Path("data/traces")
MASTER_FILES = [TRACES_DIR / "master_traces.jsonl"]
TOOLS_FILE = Path("data/tools.json")
SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")
BENCHMARK_CACHE_FILE = Path(".benchmark_cache.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    trace: Trace
    score: float
    inputs: dict
    text: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_messages(v1: dict, system_prompt: str | None) -> list[dict]:
    """Convert a v1 trace dict to OpenAI-format messages for Harmony rendering."""
    traj = v1.get("trajectory", {})
    inputs = v1.get("inputs", {})

    python_code = inputs.get("python_code", "")
    func_name = inputs.get("func_name", "")
    test_cases = inputs.get("test_cases", [])

    # Generate test assertion code inline (only consumer)
    test_lines = []
    for tc in test_cases or []:
        if isinstance(tc, (list, tuple)) and len(tc) >= 1:
            args = tc[0] if isinstance(tc[0], (list, tuple)) else tc
            args_str = ", ".join(repr(a) for a in args)
            test_lines.append(f"py.{func_name}({args_str}) == cy.{func_name}({args_str})")
    test_code = "\n".join(test_lines)

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    user_content = "\n".join(
        f"{k}: {v}" for k, v in inputs.items() if k not in ("test_cases", "benchmark_args")
    )
    msgs.append({"role": "user", "content": user_content})

    for i in range(v1.get("num_iterations", 0)):
        tool_name = traj.get(f"tool_name_{i}")
        tool_args = traj.get(f"tool_args_{i}", {})
        observation = traj.get(f"observation_{i}", "")

        if not tool_name:
            continue

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
        if isinstance(tool_args, dict) and "code" in tool_args:
            tool_args = {
                "code": tool_args["code"],
                "python_code": python_code,
                "test_code": test_code,
            }

        # Prefer reasoning_content, fall back to thought
        reasoning = traj.get(f"reasoning_{i}", "")
        if reasoning:
            reasoning = reasoning.replace("<think>\n", "").replace("\n</think>", "").strip()
        if not reasoning:
            reasoning = traj.get(f"thought_{i}", "")

        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args if tool_args else {},
                    },
                }
            ],
        }
        if reasoning:
            assistant_msg["content"] = reasoning

        msgs.append(assistant_msg)
        msgs.append(
            {"role": "tool", "tool_call_id": f"call_{i}", "name": tool_name, "content": observation}
        )

    return msgs


def render(trace: Trace, inputs: dict, system_prompt, tokenizer, tools) -> str:
    """Render a trace to Harmony text."""
    v1 = trace.to_v1_dict()
    v1["inputs"] = inputs
    msgs = build_messages(v1, system_prompt)
    return tokenizer.apply_chat_template(
        msgs,
        tools=tools,
        tokenize=False,
        reasoning_effort="medium",
    )


def assign_effort_terciles(records: list[dict]) -> None:
    """Assign reasoning effort by analysis length terciles, patch text in-place."""
    indexed = sorted(enumerate(records), key=lambda x: total_analysis_length(x[1]["text"]))
    n = len(indexed)
    t1, t2 = n // 3, 2 * n // 3

    for rank, (idx, _) in enumerate(indexed):
        effort = "low" if rank < t1 else ("medium" if rank < t2 else "high")
        records[idx]["reasoning_effort"] = effort
        records[idx]["text"] = records[idx]["text"].replace(
            "Reasoning: medium", f"Reasoning: {effort}"
        )


def load_benchmark_speedup_refs() -> dict[str, float]:
    """Load benchmark speedup references by benchmark/problem stem.

    Returns:
        Dict mapping benchmark_id/stem -> speedup reference (>1.0).
    """
    if not BENCHMARK_CACHE_FILE.exists():
        return {}

    try:
        payload = json.loads(BENCHMARK_CACHE_FILE.read_text())
    except Exception:
        logger.warning(f"Failed to parse {BENCHMARK_CACHE_FILE}; ignoring benchmark refs")
        return {}

    refs: dict[str, float] = {}
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return refs

    for bench_id, entries in results.items():
        if not isinstance(bench_id, str) or not isinstance(entries, list):
            continue
        best = 0.0
        for rec in entries:
            if not isinstance(rec, dict):
                continue
            try:
                sp = float(rec.get("speedup", 0.0) or 0.0)
            except (TypeError, ValueError):
                sp = 0.0
            if sp > best:
                best = sp
        if best > 1.0:
            refs[bench_id] = min(100.0, best)

    return refs


def trace_reasoning_char_count(trace: Trace) -> int:
    """Count reasoning characters directly from trace trajectory fields.

    Uses `reasoning_i` first, then falls back to `thought_i`, mirroring render logic.
    """
    v1 = trace.to_v1_dict()
    traj = v1.get("trajectory", {})
    total = 0
    for i in range(v1.get("num_iterations", 0)):
        reasoning = (
            (traj.get(f"reasoning_{i}", "") or "")
            .replace("<think>\n", "")
            .replace("\n</think>", "")
        )
        reasoning = reasoning.strip()
        if not reasoning:
            reasoning = (traj.get(f"thought_{i}", "") or "").strip()
        total += len(reasoning)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from trace files")
    parser.add_argument("--inputs", "-i", nargs="+", default=None)
    parser.add_argument("--output", "-o", default="data/sft_dataset.jsonl")
    parser.add_argument("--min-score", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tools", default=str(TOOLS_FILE))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--min-iters", type=int, default=0)
    parser.add_argument("--require-finish", action="store_true")
    parser.add_argument(
        "--min-analysis-chars",
        type=int,
        default=256,
        help="Minimum total analysis characters required per example (default: 256)",
    )
    parser.add_argument(
        "--allow-no-analysis",
        action="store_true",
        help="Allow rendered examples with zero analysis content (default: reject for format consistency)",
    )
    args = parser.parse_args()

    # --- Load everything up front ---
    input_paths = args.inputs or [str(p) for p in MASTER_FILES if p.exists()]

    logger.info("Loading problems...")
    problems = {p.problem_id: p for p in discover_pairs()}
    logger.info(f"  {len(problems)} problems loaded")

    tools = None
    tools_path = Path(args.tools)
    if tools_path.exists():
        tools = json.loads(tools_path.read_text())
        logger.info(f"Loaded {len(tools)} tool schemas from {tools_path}")

    system_prompt = None
    if SYSTEM_PROMPT_FILE.exists():
        system_prompt = SYSTEM_PROMPT_FILE.read_text().strip()
        logger.info(f"Loaded system prompt ({len(system_prompt)} chars)")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    logger.info("Loading traces...")
    traces = load_traces(input_paths)
    logger.info(f"Total: {len(traces)} traces")

    # Per-problem speedup normalization references.
    # Preference order:
    #   1) benchmark cache baseline by problem stem (min(100, benchmark_speedup))
    #   2) max speedup seen in traces for that problem (min(100, max_trace_speedup))
    #   3) fallback 100
    problem_max_speedup: dict[str, float] = defaultdict(float)
    for trace in traces:
        m = parse_trace_metrics(trace)
        if m["speedup"] > problem_max_speedup[trace.problem_id]:
            problem_max_speedup[trace.problem_id] = m["speedup"]

    benchmark_refs_by_stem = load_benchmark_speedup_refs()
    if benchmark_refs_by_stem:
        logger.info(
            f"Loaded benchmark refs for {len(benchmark_refs_by_stem)} problems from {BENCHMARK_CACHE_FILE}"
        )
    else:
        logger.info("No benchmark speedup refs found; using trace-derived refs")

    problem_speedup_ref: dict[str, float] = {}
    for pid in problems:
        stem = pid.split("/")[-1]
        bench_ref = benchmark_refs_by_stem.get(stem, 0.0)
        if bench_ref > 1.0:
            problem_speedup_ref[pid] = bench_ref
            continue
        mx = problem_max_speedup.get(pid, 0.0)
        problem_speedup_ref[pid] = min(100.0, mx) if mx > 1.0 else 100.0

    # --- Score & filter ---
    logger.info("Scoring and filtering traces...")
    candidates: list[Candidate] = []
    skip_problem, skip_finish, skip_iters, skip_analysis_raw = 0, 0, 0, 0

    for trace in traces:
        if (not args.allow_no_analysis) and trace_reasoning_char_count(
            trace
        ) < args.min_analysis_chars:
            skip_analysis_raw += 1
            continue
        speedup_ref = problem_speedup_ref.get(trace.problem_id, 100.0)
        sft = score_trace(trace, speedup_ref=speedup_ref)
        if sft < args.min_score:
            continue
        if args.require_finish and not any(s.tool_name == "finish" for s in trace.steps):
            skip_finish += 1
            continue
        if args.min_iters > 0:
            n_evals = sum(1 for s in trace.steps if s.tool_name != "finish")
            if n_evals < args.min_iters and not (n_evals == 1 and sft >= 0.9):
                skip_iters += 1
                continue

        problem = problems.get(trace.problem_id)
        if problem is None:
            skip_problem += 1
            continue

        trace.reward = sft
        inputs = {
            "python_code": problem.python_code,
            "func_name": problem.func_name,
            "description": problem.description or "",
            "test_cases": problem.test_cases,
            "benchmark_args": problem.benchmark_args,
        }
        candidates.append(Candidate(trace=trace, score=sft, inputs=inputs))

    logger.info(
        f"  {len(candidates)} passed "
        f"(skipped {skip_problem} unknown, {skip_finish} no finish, "
        f"{skip_iters} too few iters, {skip_analysis_raw} low/no analysis)"
    )
    logger.info(
        f"  Coverage after score/filter: {len({c.trace.problem_id for c in candidates})} problems"
    )

    # --- Validate trace structure ---
    valid = []
    n_invalid = 0
    for c in candidates:
        errs = validate_trace_for_rendering(c.trace.to_v1_dict())
        if errs:
            n_invalid += 1
        else:
            valid.append(c)
    if n_invalid:
        logger.info(f"  Trace validation: dropped {n_invalid}")
    candidates = valid
    logger.info(
        f"  Coverage after trace validation: {len({c.trace.problem_id for c in candidates})} problems"
    )

    # --- Render & token screen ---
    screened = []
    n_too_long = 0
    n_no_analysis = 0
    for c in candidates:
        c.text = render(c.trace, c.inputs, system_prompt, tokenizer, tools)
        if count_tokens(c.text) > args.max_tokens:
            n_too_long += 1
        elif (not args.allow_no_analysis) and total_analysis_length(
            c.text
        ) < args.min_analysis_chars:
            n_no_analysis += 1
        else:
            screened.append(c)
    if n_too_long:
        logger.info(f"  Token screen (>{args.max_tokens}): dropped {n_too_long}")
    if n_no_analysis:
        logger.info(
            f"  Analysis screen (<{args.min_analysis_chars} chars): dropped {n_no_analysis}"
        )
    candidates = screened
    logger.info(
        f"  Coverage after render screens: {len({c.trace.problem_id for c in candidates})} problems"
    )

    # --- Select top-k per problem (best per model, then top-k by score) ---
    by_key: dict[tuple[str, str], Candidate] = {}
    for c in candidates:
        key = (c.trace.problem_id, c.trace.model)
        if key not in by_key or c.score > by_key[key].score:
            by_key[key] = c

    pool: dict[str, list[Candidate]] = defaultdict(list)
    for (pid, _), c in by_key.items():
        pool[pid].append(c)

    selected: list[Candidate] = []
    for pid in sorted(pool):
        ranked = sorted(pool[pid], key=lambda c: c.score, reverse=True)
        selected.extend(ranked[: args.top_k])

    logger.info(f"Top-{args.top_k} selection: {len(selected)} examples from {len(pool)} problems")

    # Per-model breakdown
    raw_counts = Counter(t.model for t in traces)
    sel_counts = Counter(c.trace.model for c in selected)
    logger.info(f"  {'Model':<35} {'Traces':>6} {'Accept':>6} {'Rate':>5}")
    for model in sorted(sel_counts, key=lambda m: raw_counts[m], reverse=True):
        s, r = sel_counts[model], raw_counts[model]
        rate = f"{s / r * 100:.0f}%" if r else "-"
        name = model.replace("openrouter/", "").replace("openai/", "")
        logger.info(f"  {name:<35} {r:>6} {s:>6} {rate:>5}")

    # --- Build records & assign effort ---
    records = []
    for c in selected:
        records.append(
            {
                "text": c.text,
                "model": c.trace.model,
                "prompt_id": c.trace.prompt_id,
                "problem_id": c.trace.problem_id,
                "func_name": c.trace.func_name,
                "category": c.trace.category,
                "difficulty": c.trace.difficulty,
                "num_iterations": c.trace.num_iterations,
                "sft_score": c.score,
                "speedup": parse_trace_metrics(c.trace).get("speedup", 0.0),
            }
        )

    assign_effort_terciles(records)

    # Log effort ranges
    effort_ranges = defaultdict(list)
    for r in records:
        effort_ranges[r["reasoning_effort"]].append(total_analysis_length(r["text"]))
    for eff in ("low", "medium", "high"):
        lens = sorted(effort_ranges.get(eff, []))
        if lens:
            logger.info(
                f"  Effort '{eff}' (n={len(lens)}): "
                f"min={lens[0]}, p50={lens[len(lens) // 2]}, max={lens[-1]}"
            )

    # --- Validate rendered ---
    valid_records = []
    n_invalid = 0
    for rec in records:
        errs = validate_rendered_example(rec["text"], rec)
        if errs:
            n_invalid += 1
            logger.warning(
                f"  Validation failed ({rec['problem_id']}): {errs[0]}"
                + (f" (+{len(errs) - 1} more)" if len(errs) > 1 else "")
            )
        else:
            valid_records.append(rec)
    if n_invalid:
        logger.info(f"  Rendered validation: dropped {n_invalid}/{len(records)}")
    records = valid_records

    # --- Stats & save ---
    logger.info(f"Selected {len(records)} examples")

    if records:
        all_tok = sorted(count_tokens(r["text"]) for r in records)

        def pct(data, p):
            return data[min(int(p / 100 * len(data)), len(data) - 1)] if data else 0

        logger.info(
            f"Tokens (n={len(all_tok)}): "
            f"p50={pct(all_tok, 50)}, p90={pct(all_tok, 90)}, "
            f"p95={pct(all_tok, 95)}, p99={pct(all_tok, 99)}, max={all_tok[-1]}"
        )

    effort_counts = Counter(r.get("reasoning_effort", "?") for r in records)
    logger.info(f"Reasoning effort: {dict(effort_counts)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    logger.info(f"Saved to {output}")

    scores = [r["sft_score"] for r in records]
    if scores:
        logger.info(
            f"SFT score: min={min(scores):.3f}, max={max(scores):.3f}, "
            f"mean={sum(scores) / len(scores):.3f}"
        )


if __name__ == "__main__":
    main()
