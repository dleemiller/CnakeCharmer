"""
Create deterministic rewrite batches from Stack DuckDB GRPO candidates.

This script is for expanding unpaired GRPO Python data from Cython sources.
It emits JSONL batch files with strict filters to avoid mistaken/non-actionable
rows from the source DB.

Example:
    uv run --no-sync python scripts/create_grpo_batches.py \
      --db scripts/utils/stack_data/stack_cython_full.duckdb \
      --batch-size 50 \
      --max-items 500
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import duckdb

DEFAULT_DB = "scripts/utils/stack_data/stack_cython_full.duckdb"
DEFAULT_OUTPUT_DIR = "data/grpo_batches"
DEFAULT_USED_BLOB_IDS = "cnake_data/unpaired/used_blob_ids.json"
DEFAULT_BATCH_SIZE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GRPO rewrite batches from DuckDB candidates"
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to DuckDB database")
    parser.add_argument("--table", default="stack_cython", help="Table name inside DuckDB")
    parser.add_argument("--split", default="grpo_candidate", help="Split label to pull from")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write batches")
    parser.add_argument(
        "--used-blob-ids",
        default=DEFAULT_USED_BLOB_IDS,
        help="JSON file with already-used blob IDs (from cnake_data/unpaired)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Items per batch"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of selected items after filtering",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--min-bytes", type=int, default=200, help="Minimum source size")
    parser.add_argument("--max-bytes", type=int, default=30000, help="Maximum source size")
    parser.add_argument(
        "--allow-generated",
        action="store_true",
        help="Include rows marked as generated (default: exclude)",
    )
    parser.add_argument(
        "--allow-vendor",
        action="store_true",
        help="Include rows marked as vendor code (default: exclude)",
    )
    parser.add_argument(
        "--no-require-cython-markers",
        action="store_true",
        help="Disable Cython-marker regex filter",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts/preview only, do not write output files",
    )
    return parser.parse_args()


def load_used_blob_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    raw = json.loads(path.read_text())
    used = raw.get("used", [])
    return {str(item.get("blob_id", "")).strip() for item in used if item.get("blob_id")}


def is_used_blob_id(blob_id: str, used_blob_ids: set[str]) -> bool:
    """Match exact or prefix hashes (used list mixes short and full IDs)."""
    if blob_id in used_blob_ids:
        return True
    for used_id in used_blob_ids:
        if blob_id.startswith(used_id) or used_id.startswith(blob_id):
            return True
    return False


def load_previously_batched_ids(out_dir: Path) -> set[str]:
    manifest = out_dir / "manifest.latest.json"
    if not manifest.exists():
        return set()
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return set()
    return set(data.get("selected_blob_ids", []))


def build_query(args: argparse.Namespace) -> tuple[str, list]:
    where = [
        "split = ?",
        "extension = 'pyx'",
        "language = 'Cython'",
        "content IS NOT NULL",
        "content NOT LIKE 'SKIPPED_%'",
        "content NOT LIKE 'ERROR:%'",
        "length_bytes BETWEEN ? AND ?",
    ]
    params: list = [args.split, args.min_bytes, args.max_bytes]

    if not args.allow_generated:
        where.append("COALESCE(is_generated, FALSE) = FALSE")
    if not args.allow_vendor:
        where.append("COALESCE(is_vendor, FALSE) = FALSE")
    if not args.no_require_cython_markers:
        where.append(
            "regexp_matches(content, '(?i)\\b(cdef|cpdef|ctypedef|cimport|nogil|prange)\\b')"
        )

    sql = f"""
        SELECT
            blob_id,
            path,
            repo_name,
            length_bytes,
            content
        FROM {args.table}
        WHERE {" AND ".join(where)}
    """
    return sql, params


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    out_dir = Path(args.output_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB not found: {db_path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    used_blob_ids = load_used_blob_ids(Path(args.used_blob_ids))
    previously_batched_ids = load_previously_batched_ids(out_dir)

    con = duckdb.connect(str(db_path), read_only=True)
    sql, params = build_query(args)
    rows = con.execute(sql, params).fetchall()
    con.close()

    # Exclude IDs already converted to unpaired or already batched.
    filtered = []
    for blob_id, path, repo_name, length_bytes, content in rows:
        bid = str(blob_id)
        if is_used_blob_id(bid, used_blob_ids) or bid in previously_batched_ids:
            continue
        filtered.append(
            {
                "blob_id": bid,
                "path": path,
                "repo_name": repo_name,
                "length_bytes": int(length_bytes),
                "content": content,
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(filtered)

    if args.max_items is not None:
        filtered = filtered[: args.max_items]

    total = len(filtered)
    num_batches = math.ceil(total / args.batch_size) if total else 0

    print(f"source_rows={len(rows)}")
    print(f"after_used_filter={total}")
    print(f"batch_size={args.batch_size}")
    print(f"num_batches={num_batches}")
    if total:
        preview = filtered[:3]
        print("preview:")
        for item in preview:
            print(
                f"  - {item['blob_id']} | {item['path']} | bytes={item['length_bytes']} | repo={item['repo_name']}"
            )

    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = out_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[str] = []
    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        batch = filtered[start:end]
        batch_name = f"batch_{batch_idx + 1:04d}.jsonl"
        batch_path = batches_dir / batch_name
        with batch_path.open("w") as f:
            for item in batch:
                f.write(json.dumps(item) + "\n")
        created_files.append(str(batch_path))

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "db": str(db_path),
        "table": args.table,
        "split": args.split,
        "filters": {
            "extension": "pyx",
            "language": "Cython",
            "min_bytes": args.min_bytes,
            "max_bytes": args.max_bytes,
            "allow_generated": args.allow_generated,
            "allow_vendor": args.allow_vendor,
            "require_cython_markers": not args.no_require_cython_markers,
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "num_selected": total,
        "used_blob_ids_source": str(Path(args.used_blob_ids)),
        "num_used_blob_ids": len(used_blob_ids),
        "num_previously_batched_ids": len(previously_batched_ids),
        "batch_files": created_files,
        "selected_blob_ids": [x["blob_id"] for x in filtered],
    }
    (out_dir / "manifest.latest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote_manifest={out_dir / 'manifest.latest.json'}")
    print(f"wrote_batches={num_batches}")


if __name__ == "__main__":
    main()
