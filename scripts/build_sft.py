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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dspy-data-module" / "src"))

from dspy_data.sft import to_sft_examples

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

    # Convert to SFT format
    examples = to_sft_examples(best, tools=tools)
    logger.info(f"Built {len(examples)} SFT examples")

    # Inject system prompt and /nothink tag
    system_prompt = None
    if SYSTEM_PROMPT_FILE.exists():
        system_prompt = SYSTEM_PROMPT_FILE.read_text().strip()
        logger.info(f"Loaded system prompt ({len(system_prompt)} chars)")
    else:
        logger.warning(f"No system prompt at {SYSTEM_PROMPT_FILE}")

    for ex, trace in zip(examples, best, strict=False):
        msgs = ex["messages"]
        # Prepend system prompt
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        # Add tools column so TRL passes tool schemas to the chat template
        if tools:
            ex["tools"] = tools
        # Add /nothink to first user message for non-thinking examples
        if not trace.get("_thinking", False):
            for msg in msgs:
                if msg.get("role") == "user":
                    msg["content"] = (msg.get("content") or "") + " /nothink"
                    break

    # Fix Harmony format: move content to "thinking" field on assistant messages with tool_calls.
    # The Harmony chat template renders "thinking" as an analysis channel message before the tool call.
    # Without this, the template silently drops the content (reasoning/thinking) from tool-call messages.
    thinking_moved = 0
    for ex in examples:
        for msg in ex["messages"]:
            if msg.get("role") == "assistant" and msg.get("tool_calls") and msg.get("content"):
                msg["thinking"] = msg.pop("content")
                thinking_moved += 1
    if thinking_moved:
        logger.info(
            f"Moved content→thinking on {thinking_moved} assistant+tool_call messages (Harmony format)"
        )

    # Patch malformed finish calls: model passed code args to finish, causing DSPy error.
    # Strip args and clear error content so the trace looks like a clean finish.
    patched = 0
    for ex in examples:
        msgs = ex["messages"]
        for i, msg in enumerate(msgs):
            if (
                msg.get("role") == "tool"
                and msg.get("name") == "finish"
                and "Execution error" in (msg.get("content") or "")
            ):
                msg["content"] = "Completed."
                asst = msgs[i - 1]
                for tc in asst.get("tool_calls", []):
                    if tc.get("function", {}).get("name") == "finish":
                        tc["function"]["arguments"] = {}
                patched += 1
    if patched:
        logger.info(f"Patched {patched} malformed finish calls")

    # Attach trace metadata to each example
    METADATA_KEYS = (
        "model",
        "prompt_id",
        "problem_id",
        "func_name",
        "category",
        "difficulty",
        "num_iterations",
    )
    for ex, trace in zip(examples, best, strict=False):
        for k in METADATA_KEYS:
            if k in trace:
                ex[k] = trace[k]
        ex["sft_score"] = trace["reward"]
        ex["speedup"] = parse_trace_metrics(trace).get("speedup", 0.0)
        ex["thinking"] = trace.get("_thinking", False)

    # Token counting and filtering
    import re as _re

    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")

    def count_tokens(ex: dict) -> int:
        return len(enc.encode(json.dumps(ex, default=str)))

    def truncate_reasoning(ex: dict, max_tok: int) -> tuple[dict, bool]:
        """Try to fit example within max_tok by truncating <think> blocks.

        Returns (example, was_truncated). Modifies example in place.
        """
        current = count_tokens(ex)
        if current <= max_tok:
            return ex, False

        # Collect (msg_index, field, reasoning_text, token_count) for all think blocks
        # Check both "content" and "thinking" fields (Harmony format moves content→thinking)
        think_blocks = []
        for i, msg in enumerate(ex["messages"]):
            if msg.get("role") != "assistant":
                continue
            for field in ("content", "thinking"):
                text = msg.get(field)
                if not text:
                    continue
                match = _re.search(r"<think>\n(.*?)\n</think>", text, _re.DOTALL)
                if match:
                    reasoning = match.group(1)
                    think_blocks.append((i, field, match, len(enc.encode(reasoning))))

        if not think_blocks:
            return ex, False

        # Sort by token count descending, truncate longest first
        think_blocks.sort(key=lambda x: x[2], reverse=True)
        excess = current - max_tok

        for msg_idx, field, match, block_tokens in think_blocks:
            if excess <= 0:
                break
            msg = ex["messages"][msg_idx]
            text = msg[field]
            if excess >= block_tokens:
                text = text[: match.start()] + text[match.end() :]
                msg[field] = text.strip() or None
                excess -= block_tokens
            else:
                reasoning = match.group(1)
                keep_tokens = block_tokens - excess
                if keep_tokens > 0:
                    truncated = enc.decode(enc.encode(reasoning)[:keep_tokens]) + "..."
                    msg[field] = (
                        text[: match.start()]
                        + f"<think>\n{truncated}\n</think>"
                        + text[match.end() :]
                    )
                else:
                    text = text[: match.start()] + text[match.end() :]
                    msg[field] = text.strip() or None
                excess = 0

        return ex, True

    # Apply token filtering
    kept = []
    n_truncated = 0
    n_dropped = 0
    for ex in examples:
        tok = count_tokens(ex)
        if tok <= args.max_tokens:
            kept.append(ex)
            continue
        ex, was_truncated = truncate_reasoning(ex, args.max_tokens)
        tok = count_tokens(ex)
        if tok <= args.max_tokens:
            kept.append(ex)
            if was_truncated:
                n_truncated += 1
        else:
            n_dropped += 1

    logger.info(
        f"Token filter (max={args.max_tokens}): kept={len(kept)}, "
        f"truncated={n_truncated}, dropped={n_dropped}"
    )
    examples = kept

    # Token distribution stats
    if examples:
        all_tok = sorted(count_tokens(ex) for ex in examples)
        think_tok = sorted(count_tokens(ex) for ex in examples if ex.get("thinking"))
        nothink_tok = sorted(count_tokens(ex) for ex in examples if not ex.get("thinking"))
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

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, default=str) + "\n")

    msg_counts = [len(ex["messages"]) for ex in examples]
    logger.info(f"Saved to {output}")
    if msg_counts:
        logger.info(
            f"Messages per example: min={min(msg_counts)}, max={max(msg_counts)}, "
            f"mean={sum(msg_counts) / len(msg_counts):.1f}"
        )

    # Score distribution
    scores = [ex.get("sft_score", 0) for ex in examples]
    if scores:
        logger.info(
            f"SFT score: min={min(scores):.3f}, max={max(scores):.3f}, "
            f"mean={sum(scores) / len(scores):.3f}"
        )


if __name__ == "__main__":
    main()
