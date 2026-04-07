"""Publish the curated py/cy/test dataset to bsmith925/cnake-charmer.

Reads all 665 problem pairs via discover_pairs() and enriches each row with
benchmark results from .benchmark_cache.json, then pushes a 'curated' config.

Usage:
    uv run python scripts/publish_dataset.py [--repo-id OWNER/REPO] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ID = "bsmith925/cnake-charmer"
REPO_ROOT = Path(__file__).parent.parent
BENCHMARK_CACHE = REPO_ROOT / ".benchmark_cache.json"


def load_benchmark_cache() -> dict:
    if not BENCHMARK_CACHE.exists():
        return {}
    data = json.loads(BENCHMARK_CACHE.read_text())
    return data.get("results", {})


def build_rows(cache: dict) -> list[dict]:
    from cnake_data.loader import discover_pairs

    problems = discover_pairs()
    rows = []
    for p in problems:
        stem = Path(p.metadata.get("py_path", "")).stem
        bench = cache.get(stem, [{}])[0] if cache.get(stem) else {}

        test_code = ""
        test_path = p.metadata.get("test_path", "")
        if test_path:
            tp = Path(test_path)
            if tp.exists():
                test_code = tp.read_text()

        rows.append(
            {
                "problem_id": p.problem_id,
                "category": p.category,
                "difficulty": p.difficulty,
                "func_name": p.func_name,
                "description": p.description,
                "python_code": p.python_code,
                "cython_code": p.cython_code,
                "test_code": test_code,
                "benchmark_args": json.dumps(list(p.benchmark_args)) if p.benchmark_args else None,
                "speedup": bench.get("speedup"),
                "py_avg_sec": bench.get("py_avg"),
                "cy_avg_sec": bench.get("cy_avg"),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Publish curated dataset to HuggingFace Hub.")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", REPO_ID),
        help=f"HuggingFace dataset repo ID (default: {REPO_ID})",
    )
    parser.add_argument(
        "--token", help="HuggingFace token (default: HF_TOKEN env or huggingface-cli login)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without pushing")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    print("Loading benchmark cache...")
    cache = load_benchmark_cache()
    print(f"  {len(cache)} benchmark results loaded")

    print("Building dataset rows...")
    rows = build_rows(cache)
    print(f"  {len(rows)} problems")

    if args.dry_run:
        print("\n--- Dry run ---")
        print(f"Columns: {list(rows[0].keys()) if rows else '(none)'}")
        print(f"Rows with speedup: {sum(1 for r in rows if r['speedup'] is not None)}")
        print(f"Rows with cython: {sum(1 for r in rows if r['cython_code'])}")
        print(f"Rows with tests: {sum(1 for r in rows if r['test_code'])}")
        if rows:
            sample = {
                k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                for k, v in rows[0].items()
            }
            print(f"\nSample row:\n{json.dumps(sample, indent=2)}")
        return

    from datasets import Dataset, DatasetDict

    ds = Dataset.from_list(rows)
    dd = DatasetDict({"train": ds})

    print(f"\nPushing 'curated' config to {args.repo_id}...")
    dd.push_to_hub(args.repo_id, config_name="curated", token=token)
    print(f"Done. Dataset live at https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
