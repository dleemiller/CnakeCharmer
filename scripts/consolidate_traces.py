"""
Consolidate all trace JSONL files into a single master file (v2 format).

Deduplicates by (problem_id, model, first_thought) fingerprint. When a trace
exists in both thinking and nothink versions, only the thinking version is kept.

After consolidation, collect_traces.py appends directly to master_traces.jsonl
and uses (problem_id, model) to determine what's already done.

Usage:
    uv run --no-sync python scripts/consolidate_traces.py
    uv run --no-sync python scripts/consolidate_traces.py --dry-run
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.traces.io import load_traces, save_traces
from cnake_charmer.traces.models import Trace

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACES_DIR = Path("data/traces")
MASTER_FILE = TRACES_DIR / "master_traces.jsonl"


def fingerprint(trace: Trace) -> tuple:
    """Unique identity of a trace: (problem_id, model, first thought content)."""
    first_thought = ""
    if trace.steps:
        first_thought = (trace.steps[0].thought or "")[:200]
    return (trace.problem_id, trace.model, first_thought)


def main():
    parser = argparse.ArgumentParser(description="Consolidate traces into master file")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Input JSONL files (default: all non-master files in data/traces/)",
    )
    args = parser.parse_args()

    # Collect input files
    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        input_paths = sorted(
            f for f in TRACES_DIR.glob("*.jsonl") if not f.name.startswith("master_")
        )

    if not input_paths:
        logger.info("No input files found — nothing to consolidate")
        return

    # Load existing master if present
    all_traces: list[Trace] = []
    if MASTER_FILE.exists():
        logger.info("Loading existing master file...")
        all_traces = load_traces([MASTER_FILE])

    # Load new traces
    if input_paths:
        logger.info("Loading input files...")
        new_traces = load_traces([str(p) for p in input_paths])
        all_traces.extend(new_traces)

    logger.info(f"Total: {len(all_traces)} traces (before dedup)")

    # Deduplicate: prefer thinking version over nothink
    seen: dict[tuple, Trace] = {}
    for trace in all_traces:
        fp = fingerprint(trace)
        if fp in seen:
            # Prefer thinking over nothink
            if trace.thinking and not seen[fp].thinking:
                seen[fp] = trace
        else:
            seen[fp] = trace

    deduped = list(seen.values())

    # Stats
    n_thinking = sum(1 for t in deduped if t.thinking)
    n_nothink = len(deduped) - n_thinking
    think_models = Counter(t.model for t in deduped if t.thinking)
    nothink_models = Counter(t.model for t in deduped if not t.thinking)
    n_problems = len({t.problem_id for t in deduped})

    logger.info(f"\nDeduplicated: {len(deduped)} unique traces, {n_problems} problems")
    logger.info(f"  thinking: {n_thinking}")
    for m, c in think_models.most_common():
        logger.info(f"    {m}: {c}")
    logger.info(f"  nothink: {n_nothink}")
    for m, c in nothink_models.most_common():
        logger.info(f"    {m}: {c}")

    if args.dry_run:
        logger.info("Dry run — not writing files")
        return

    # Write master file
    count = save_traces(deduped, MASTER_FILE)
    logger.info(f"\nWrote {count} traces to {MASTER_FILE}")

    # Move consumed input files to archive
    if not args.inputs:
        # Only archive auto-discovered files, not explicitly passed ones
        archive = TRACES_DIR / "archived"
        archive.mkdir(exist_ok=True)
        for f in input_paths:
            f.rename(archive / f.name)
            logger.info(f"  Archived {f.name}")


if __name__ == "__main__":
    main()
