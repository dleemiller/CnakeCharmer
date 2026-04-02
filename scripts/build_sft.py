"""
Build SFT dataset from collected trace files using sft_scoring for selection.

Loads traces from data/traces/*.jsonl, re-scores with sft_scoring.score_trace(),
augments with problem inputs from discover_pairs(), selects best per problem,
and converts to messages format via dspy_data.sft.

Harmony format handling:
- Moves assistant content to "thinking" field when tool_calls are present,
  so the chat template renders it as an analysis channel message.
- Includes tools column so TRL passes tool schemas to the chat template.

Usage:
    uv run --no-sync python scripts/build_sft.py --require-finish --min-iters 2
    uv run --no-sync python scripts/build_sft.py --min-score 0.5 --output data/sft_dataset.jsonl
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.dataset.loader import discover_pairs
from cnake_charmer.training.sft_scoring import parse_trace_metrics, score_trace

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACES_DIR = Path("data/traces")
# master_nothink.jsonl excluded: all models there were thinking models with uncaptured reasoning
MASTER_FILES = [TRACES_DIR / "master_thinking.jsonl"]
TOOLS_FILE = Path("data/tools.json")
SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")


def load_traces(paths: list[str]) -> list[dict]:
    traces = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"Skipping missing file: {path}")
            continue
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                        count += 1
                    except json.JSONDecodeError:
                        pass
        logger.info(f"  {path.name}: {count} traces")
    return traces


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from trace files")
    parser.add_argument(
        "--inputs",
        "-i",
        nargs="+",
        default=None,
        help="Input JSONL trace files (default: all data/traces/*.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/sft_dataset.jsonl",
        help="Output JSONL path (default: data/sft_dataset.jsonl)",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.8, help="Minimum sft_score to include (default: 0.8)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Keep top-k traces per problem above threshold (default: 1)",
    )
    parser.add_argument(
        "--tools", default=str(TOOLS_FILE), help=f"Tool schema JSON file (default: {TOOLS_FILE})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens per example; truncate reasoning then drop (default: 16384)",
    )
    parser.add_argument(
        "--min-iters",
        type=int,
        default=0,
        help="Minimum tool-call iterations to include (default: 0, no filter)",
    )
    parser.add_argument(
        "--require-finish",
        action="store_true",
        help="Only include traces that called finish (exclude force-exits)",
    )
    args = parser.parse_args()

    input_paths = args.inputs or [str(p) for p in MASTER_FILES if p.exists()]

    # Load problems for inputs augmentation
    logger.info("Loading problems...")
    problems = {p.problem_id: p for p in discover_pairs()}
    logger.info(f"  {len(problems)} problems loaded")

    # Load tools
    tools = None
    tools_path = Path(args.tools)
    if tools_path.exists():
        tools = json.loads(tools_path.read_text())
        logger.info(f"Loaded {len(tools)} tool schemas from {tools_path}")
    else:
        logger.warning(f"No tools file at {tools_path} — 'tools' column will be omitted")

    # Load all traces
    logger.info("Loading traces...")
    traces = load_traces(input_paths)
    logger.info(f"Total: {len(traces)} traces")

    # Score and augment with inputs
    logger.info("Scoring and augmenting traces...")
    scored = []
    skipped_no_problem = 0
    skipped_no_finish = 0
    skipped_min_iters = 0
    for trace in traces:
        sft = score_trace(trace)
        if sft < args.min_score:
            continue

        # Filter: require finish tool call (exclude force-exits that hit max iterations)
        traj = trace.get("trajectory", {})
        if args.require_finish:
            called_finish = any(v == "finish" for k, v in traj.items() if k.startswith("tool_name"))
            if not called_finish:
                skipped_no_finish += 1
                continue

        # Filter: minimum iterations (eval calls excluding finish)
        # 1-eval traces are allowed if they scored very well (annotation >0.9, speedup >2x)
        if args.min_iters > 0:
            n_evals = sum(1 for k, v in traj.items() if k.startswith("tool_name") and v != "finish")
            if n_evals < args.min_iters:
                # Allow high-quality 1-shot traces through
                if n_evals == 1 and sft >= 0.9:
                    pass  # keep it — model nailed it first try
                else:
                    skipped_min_iters += 1
                    continue

        pid = trace.get("problem_id", "")
        problem = problems.get(pid)
        if problem is None:
            skipped_no_problem += 1
            continue

        trace["reward"] = sft
        trace["inputs"] = {
            "python_code": problem.python_code,
            "func_name": problem.func_name,
            "description": problem.description or "",
            "test_cases": problem.test_cases,
            "benchmark_args": problem.benchmark_args,
        }
        scored.append(trace)

    logger.info(
        f"  {len(scored)} traces passed (skipped {skipped_no_problem} unknown problem_ids, "
        f"{skipped_no_finish} no finish, {skipped_min_iters} too few iters)"
    )

    # Classify thinking vs non-thinking
    def has_thinking(trace):
        return any(k.startswith("reasoning_") for k in trace.get("trajectory", {}))

    # Step 1: best per (problem, model, thinking) — deduplicate
    by_key = defaultdict(list)
    for trace in scored:
        key = (trace["problem_id"], trace.get("model", "unknown"), has_thinking(trace))
        by_key[key].append(trace)

    best_per_key = {}
    for key, candidates in by_key.items():
        best_per_key[key] = max(candidates, key=lambda t: t["reward"])

    # Step 2: top-k per problem per thinking split, unique models
    thinking_pool = defaultdict(list)  # pid -> [traces with reasoning]
    nothink_pool = defaultdict(list)  # pid -> [traces without reasoning]
    for (pid, _model, is_thinking), trace in best_per_key.items():
        if is_thinking:
            thinking_pool[pid].append(trace)
        else:
            nothink_pool[pid].append(trace)

    best = []
    all_pids = sorted(set(list(thinking_pool.keys()) + list(nothink_pool.keys())))
    for pid in all_pids:
        for pool in (thinking_pool, nothink_pool):
            candidates = pool.get(pid, [])
            ranked = sorted(candidates, key=lambda t: t["reward"], reverse=True)
            best.extend(ranked[: args.top_k])

    # Tag thinking
    for trace in best:
        trace["_thinking"] = has_thinking(trace)

    problems_covered = len(all_pids)
    n_thinking = sum(1 for t in best if t["_thinking"])
    n_nothink = len(best) - n_thinking
    logger.info(
        f"Top-{args.top_k} selection: {len(best)} examples from {problems_covered} problems"
    )
    logger.info(f"  thinking={n_thinking}, nothink={n_nothink}")

    # Per-model breakdown
    raw_counts = Counter(t.get("model", "?") for t in traces)
    sel_models = defaultdict(lambda: {"total": 0, "think": 0, "nothink": 0})
    for t in best:
        m = t.get("model", "?")
        sel_models[m]["total"] += 1
        sel_models[m]["think" if t["_thinking"] else "nothink"] += 1

    logger.info("Per-model breakdown:")
    logger.info(
        f"  {'Model':<35} {'Traces':>6} {'Accept':>6} {'Rate':>5} {'Think':>5} {'NoThk':>5}"
    )
    for model in sorted(sel_models, key=lambda m: raw_counts[m], reverse=True):
        s = sel_models[model]
        r = raw_counts[model]
        rate = f"{s['total'] / r * 100:.0f}%" if r > 0 else "-"
        name = model.replace("openrouter/", "").replace("openai/", "")
        logger.info(
            f"  {name:<35} {r:>6} {s['total']:>6} {rate:>5} {s['think']:>5} {s['nothink']:>5}"
        )

    # Load system prompt and tokenizer for Harmony rendering
    system_prompt = None
    if SYSTEM_PROMPT_FILE.exists():
        system_prompt = SYSTEM_PROMPT_FILE.read_text().strip()
        logger.info(f"Loaded system prompt ({len(system_prompt)} chars)")
    else:
        logger.warning(f"No system prompt at {SYSTEM_PROMPT_FILE}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    # Compute median reasoning length for thinking traces to set medium/high threshold
    reasoning_lengths = []
    for trace in best:
        if not trace.get("_thinking"):
            continue
        traj = trace.get("trajectory", {})
        total = sum(
            len(v) for k, v in traj.items() if k.startswith("reasoning_") and isinstance(v, str)
        )
        if total > 0:
            reasoning_lengths.append(total)
    median_reasoning = (
        sorted(reasoning_lengths)[len(reasoning_lengths) // 2] if reasoning_lengths else 0
    )
    logger.info(f"Median reasoning length: {median_reasoning} chars (for medium/high threshold)")

    # Build messages directly from traces and render to Harmony text.
    # Key design decisions for Harmony format:
    # - Thinking traces use reasoning_N for analysis channel (deep reasoning) → medium/high effort
    # - Nothink traces use thought_N for analysis channel (brief planning) → low effort
    # - NO final standalone code message — end with finish tool call so all analysis channels render
    #   (Harmony template drops analysis when a future non-tool-call assistant message exists)
    rendered = []
    for trace in best:
        traj = trace.get("trajectory", {})
        is_thinking = trace.get("_thinking", False)
        inputs = trace.get("inputs", {})

        # Determine reasoning effort
        if not is_thinking:
            effort = "low"
        else:
            total_reasoning = sum(
                len(v) for k, v in traj.items() if k.startswith("reasoning_") and isinstance(v, str)
            )
            effort = "high" if total_reasoning > median_reasoning else "medium"

        # Build messages
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})

        # User message
        user_content = "\n".join(
            f"{k}: {v}" for k, v in inputs.items() if k not in ("test_cases", "benchmark_args")
        )
        msgs.append({"role": "user", "content": user_content})

        # Tool-calling turns from trajectory
        n_iters = trace.get("num_iterations", 0)
        for i in range(n_iters):
            tool_name = traj.get(f"tool_name_{i}")
            tool_args = traj.get(f"tool_args_{i}", {})
            observation = traj.get(f"observation_{i}", "")

            if not tool_name:
                continue

            # Select reasoning for analysis channel:
            # - thinking traces: use reasoning_N (deep <think> content, stripped of tags)
            # - nothink traces: use thought_N (brief ReAct planning)
            if is_thinking:
                reasoning = traj.get(f"reasoning_{i}", "")
                # Strip <think> tags if present
                reasoning = reasoning.replace("<think>\n", "").replace("\n</think>", "").strip()
                if not reasoning:
                    reasoning = traj.get(f"thought_{i}", "")
            else:
                reasoning = traj.get(f"thought_{i}", "")

            # Assistant message with tool call + thinking for analysis channel
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

            # Patch malformed finish calls
            if tool_name == "finish" and "Execution error" in observation:
                observation = "Completed."
                assistant_msg["tool_calls"][0]["function"]["arguments"] = {}

            msgs.append(assistant_msg)
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "name": tool_name,
                    "content": observation,
                }
            )

        # NO final standalone assistant message — this ensures all analysis channels render

        # Render to Harmony text
        text = tokenizer.apply_chat_template(
            msgs,
            tools=tools,
            tokenize=False,
            reasoning_effort=effort,
        )

        # Attach metadata
        record = {"text": text}
        for k in (
            "model",
            "prompt_id",
            "problem_id",
            "func_name",
            "category",
            "difficulty",
            "num_iterations",
        ):
            if k in trace:
                record[k] = trace[k]
        record["sft_score"] = trace["reward"]
        record["speedup"] = parse_trace_metrics(trace).get("speedup", 0.0)
        record["thinking"] = is_thinking
        record["reasoning_effort"] = effort
        rendered.append(record)

    examples = rendered
    logger.info(f"Rendered {len(examples)} examples to Harmony text")

    # Token counting using the actual tokenizer
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Filter by token count
    kept = []
    n_dropped = 0
    for ex in examples:
        tok = count_tokens(ex["text"])
        if tok <= args.max_tokens:
            kept.append(ex)
        else:
            n_dropped += 1

    logger.info(f"Token filter (max={args.max_tokens}): kept={len(kept)}, dropped={n_dropped}")
    examples = kept

    # Token distribution stats
    if examples:
        all_tok = sorted(count_tokens(ex["text"]) for ex in examples)
        think_tok = sorted(count_tokens(ex["text"]) for ex in examples if ex.get("thinking"))
        nothink_tok = sorted(count_tokens(ex["text"]) for ex in examples if not ex.get("thinking"))
        n = len(all_tok)

        def pct(data, p):
            return data[min(int(p / 100 * len(data)), len(data) - 1)] if data else 0

        logger.info(
            f"Tokens (all n={n}): "
            f"p50={pct(all_tok, 50)}, p90={pct(all_tok, 90)}, "
            f"p95={pct(all_tok, 95)}, p99={pct(all_tok, 99)}, max={all_tok[-1]}"
        )
        if think_tok:
            logger.info(
                f"Tokens (thinking n={len(think_tok)}): "
                f"p50={pct(think_tok, 50)}, p90={pct(think_tok, 90)}, "
                f"p95={pct(think_tok, 95)}, max={think_tok[-1]}"
            )
        if nothink_tok:
            logger.info(
                f"Tokens (nothink n={len(nothink_tok)}): "
                f"p50={pct(nothink_tok, 50)}, p90={pct(nothink_tok, 90)}, "
                f"p95={pct(nothink_tok, 95)}, max={nothink_tok[-1]}"
            )

    # Reasoning effort distribution
    effort_counts = Counter(ex.get("reasoning_effort", "?") for ex in examples)
    logger.info(f"Reasoning effort: {dict(effort_counts)}")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, default=str) + "\n")

    logger.info(f"Saved to {output}")

    # Score distribution
    scores = [ex.get("sft_score", 0) for ex in examples]
    if scores:
        logger.info(
            f"SFT score: min={min(scores):.3f}, max={max(scores):.3f}, "
            f"mean={sum(scores) / len(scores):.3f}"
        )


if __name__ == "__main__":
    main()
