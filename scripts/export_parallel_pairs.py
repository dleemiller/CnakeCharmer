"""Export paired Python/Cython implementations with benchmark-cache metadata.

Reads benchmark results from .benchmark_cache.json and writes one JSONL record
per discovered Python/Cython pair. This avoids expensive compile/test/ASan work
and is intended to be fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from cnake_data.loader import discover_pairs

logger = logging.getLogger(__name__)


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_benchmark_cache(path: Path) -> dict:
    if not path.exists():
        logger.warning("Benchmark cache not found: %s", path)
        return {"hashes": {}, "results": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("cache root is not an object")
        data.setdefault("results", {})
        return data
    except Exception as exc:
        logger.warning("Failed to read benchmark cache %s: %s", path, exc)
        return {"hashes": {}, "results": {}}


def _pick_benchmark_entry(entries: list[dict]) -> dict | None:
    if not entries:
        return None
    # Prefer canonical 'cython'. Otherwise choose the highest-speedup variant.
    for e in entries:
        if e.get("syntax") == "cython":
            return e
    return max(entries, key=lambda e: float(e.get("speedup", 0.0)))


def build_record(problem, speedup: float | None) -> dict:
    return {
        "problem_id": problem.problem_id,
        "category": problem.category,
        "difficulty": problem.difficulty,
        "func_name": problem.func_name,
        "description": problem.description,
        "speedup": speedup,
        "python_code": problem.python_code,
        "cython_code": problem.cython_code,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paired Python/Cython implementations to JSONL"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/hf/parallel/parallel_examples.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--cache-file",
        default=".benchmark_cache.json",
        help="Path to benchmark cache JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of pairs to export",
    )
    parser.add_argument(
        "--sandbox-events",
        action="store_true",
        default=False,
        help="Show verbose sandbox lifecycle logs",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    os.environ["CNAKE_SANDBOX_LOG_EVENTS"] = "1" if args.sandbox_events else "0"
    if not args.sandbox_events:
        logging.getLogger("cnake_charmer.eval.sandbox").setLevel(logging.WARNING)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_benchmark_cache(Path(args.cache_file))
    cache_results = cache.get("results", {})

    problems = [p for p in discover_pairs() if p.has_python and p.has_cython]
    if args.limit is not None:
        problems = problems[: max(0, args.limit)]

    logger.info("Exporting %d paired implementations -> %s", len(problems), output_path)

    exported = 0
    cache_hits = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, problem in enumerate(problems, start=1):
            benchmark_id = problem.problem_id.split("/", 1)[-1]
            bench_entry = _pick_benchmark_entry(cache_results.get(benchmark_id, []))
            speedup = (
                float(bench_entry.get("speedup", 0.0))
                if isinstance(bench_entry, dict) and bench_entry.get("speedup") is not None
                else None
            )
            record = build_record(problem, speedup)
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            f.flush()

            exported += 1
            if bench_entry is not None:
                cache_hits += 1
            speedup_str = f"{speedup:.2f}x" if isinstance(speedup, float) else "-"
            logger.info(
                "[%d/%d] %s cache=%s speedup=%s",
                idx,
                len(problems),
                problem.problem_id,
                "hit" if bench_entry is not None else "miss",
                speedup_str,
            )

    logger.info(
        "Done. Exported=%d cache_hits=%d (%.1f%%)",
        exported,
        cache_hits,
        (100.0 * cache_hits / exported) if exported else 0.0,
    )


if __name__ == "__main__":
    main()
