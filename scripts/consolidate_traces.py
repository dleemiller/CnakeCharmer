"""
Consolidate all trace JSONL files into two master files: thinking and nothink.

Deduplicates by (problem_id, model, thought_0) fingerprint. When a trace
exists in both thinking (has reasoning_N) and nothink versions, only the
thinking version is kept.

After consolidation, collect_traces.py appends directly to the master files
and uses (problem_id, model) to determine what's already done.

Usage:
    uv run --no-sync python scripts/consolidate_traces.py
    uv run --no-sync python scripts/consolidate_traces.py --dry-run
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACES_DIR = Path("data/traces")
ARCHIVE_DIR = Path("data/traces_old")
THINKING_FILE = TRACES_DIR / "master_thinking.jsonl"
NOTHINK_FILE = TRACES_DIR / "master_nothink.jsonl"


def fingerprint(trace):
    """Unique identity of a trace: (problem_id, model, first thought content)."""
    traj = trace.get("trajectory", {})
    return (
        trace.get("problem_id", ""),
        trace.get("model", ""),
        (traj.get("thought_0") or "")[:200],
    )


def has_thinking(trace):
    return any(k.startswith("reasoning_") for k in trace.get("trajectory", {}))


def load_all_traces():
    """Load all traces from data/traces/ and data/traces/superseded/."""
    traces = []
    dirs = [TRACES_DIR]
    if (TRACES_DIR / "superseded").exists():
        dirs.append(TRACES_DIR / "superseded")

    for d in dirs:
        for f in sorted(d.glob("*.jsonl")):
            if f.name.startswith("master_"):
                continue
            count = 0
            with open(f) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        traces.append((json.loads(line), f.name))
                        count += 1
                    except json.JSONDecodeError:
                        pass
            logger.info(f"  {f.name}: {count}")
    return traces


def main():
    parser = argparse.ArgumentParser(description="Consolidate traces into master files")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    args = parser.parse_args()

    logger.info("Loading all traces...")
    all_traces = load_all_traces()
    logger.info(f"Total: {len(all_traces)} traces")

    # Deduplicate: prefer thinking version over nothink
    seen = {}  # fingerprint -> (trace, is_thinking)
    for trace, _source in all_traces:
        fp = fingerprint(trace)
        thinking = has_thinking(trace)

        if fp in seen:
            existing_thinking = seen[fp][1]
            # Prefer thinking over nothink
            if thinking and not existing_thinking:
                seen[fp] = (trace, thinking)
            # Otherwise keep existing (first seen)
        else:
            seen[fp] = (trace, thinking)

    thinking_traces = []
    nothink_traces = []
    for _fp, (trace, is_thinking) in seen.items():
        if is_thinking:
            thinking_traces.append(trace)
        else:
            nothink_traces.append(trace)

    # Stats
    from collections import Counter

    think_models = Counter(t.get("model", "?") for t in thinking_traces)
    nothink_models = Counter(t.get("model", "?") for t in nothink_traces)
    think_problems = len({t.get("problem_id") for t in thinking_traces})
    nothink_problems = len({t.get("problem_id") for t in nothink_traces})

    logger.info(f"\nDeduplicated: {len(seen)} unique traces")
    logger.info(f"  thinking: {len(thinking_traces)} traces, {think_problems} problems")
    for m, c in think_models.most_common():
        logger.info(f"    {m}: {c}")
    logger.info(f"  nothink:  {len(nothink_traces)} traces, {nothink_problems} problems")
    for m, c in nothink_models.most_common():
        logger.info(f"    {m}: {c}")

    if args.dry_run:
        logger.info("Dry run — not writing files")
        return

    # Write master files
    for path, traces in [(THINKING_FILE, thinking_traces), (NOTHINK_FILE, nothink_traces)]:
        with open(path, "w") as f:
            for t in traces:
                f.write(json.dumps(t, default=str) + "\n")
        logger.info(f"Wrote {len(traces)} traces to {path}")

    # Move old files to archive
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(TRACES_DIR.glob("*.jsonl")):
        if f.name.startswith("master_"):
            continue
        f.rename(ARCHIVE_DIR / f.name)
        logger.info(f"  Archived {f.name}")

    # Clean up superseded
    superseded = TRACES_DIR / "superseded"
    if superseded.exists():
        for f in superseded.glob("*.jsonl"):
            f.rename(ARCHIVE_DIR / f"superseded_{f.name}")
            logger.info(f"  Archived superseded/{f.name}")
        superseded.rmdir()

    logger.info("\nDone. Active files:")
    logger.info(f"  {THINKING_FILE}")
    logger.info(f"  {NOTHINK_FILE}")
    logger.info(f"  collect_traces.py will append new traces to {THINKING_FILE}")


if __name__ == "__main__":
    main()
