"""One-time seed: upload Stack v2 corpus to bsmith925/cnake-charmer-stack-v2.

Pushes two dataset configs:
  - full       : all 21,799 rows from stack_cython_full.duckdb (table stack_cython)
  - candidates : 700 sft_candidate rows from sft_candidates.duckdb (table sft_data)

Also initializes an empty data/checkouts.jsonl registry on the HF repo.

After a successful run, remove the DuckDB files from git tracking:
    git rm --cached utils/stack_data/stack_cython_full.duckdb sft_candidates.duckdb

Usage:
    uv run python scripts/upload_corpus.py [--config {full,candidates,all}] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ID = "bsmith925/cnake-charmer-stack-v2"
FULL_DB = Path(__file__).parent.parent / "utils" / "stack_data" / "stack_cython_full.duckdb"
CANDIDATES_DB = Path(__file__).parent.parent / "sft_candidates.duckdb"


def load_full(dry_run: bool):
    import duckdb
    from datasets import Dataset, DatasetDict

    print(f"Loading full corpus from {FULL_DB}...")
    conn = duckdb.connect(str(FULL_DB), read_only=True)
    rows = conn.execute(
        """
        SELECT blob_id, filename, path, repo_name, length_bytes, src_encoding,
               content, split, detected_licenses, license_type, star_events_count,
               extension, language, is_generated, is_vendor
        FROM stack_cython
        """
    ).fetchdf()
    conn.close()

    print(f"  {len(rows):,} rows loaded")
    if dry_run:
        print(f"  Columns: {list(rows.columns)}")
        print(f"  Sample blob_id: {rows['blob_id'].iloc[0]}")
        return None

    ds = Dataset.from_pandas(rows, preserve_index=False)
    return DatasetDict({"train": ds})


def load_candidates(dry_run: bool):
    import duckdb
    from datasets import Dataset, DatasetDict

    print(f"Loading candidates from {CANDIDATES_DB}...")
    conn = duckdb.connect(str(CANDIDATES_DB), read_only=True)
    rows = conn.execute(
        """
        SELECT blob_id, filename, path, repo_name, length_bytes, src_encoding,
               content, split, detected_licenses, license_type, star_events_count
        FROM sft_data
        WHERE split = 'sft_candidate'
        """
    ).fetchdf()
    conn.close()

    print(f"  {len(rows):,} rows loaded")
    if dry_run:
        print(f"  Columns: {list(rows.columns)}")
        print(f"  Sample blob_id: {rows['blob_id'].iloc[0]}")
        return None

    ds = Dataset.from_pandas(rows, preserve_index=False)
    return DatasetDict({"train": ds})


def main():
    parser = argparse.ArgumentParser(description="Upload Stack v2 corpus to HuggingFace Hub.")
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {REPO_ID})",
    )
    parser.add_argument(
        "--token", help="HuggingFace token (default: HF_TOKEN env or huggingface-cli login)"
    )
    parser.add_argument(
        "--config",
        choices=["full", "candidates", "all"],
        default="all",
        help="Which config to upload (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without pushing")
    args = parser.parse_args()

    import os

    token = args.token or os.environ.get("HF_TOKEN")

    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, token=token)
        print(f"Repo {args.repo_id} ready.")

    upload_full = args.config in ("full", "all")
    upload_candidates = args.config in ("candidates", "all")

    if upload_full:
        dd = load_full(args.dry_run)
        if not args.dry_run:
            print(f"Pushing 'full' config to {args.repo_id}...")
            dd.push_to_hub(args.repo_id, config_name="full", token=token)
            print("  Done.")

    if upload_candidates:
        dd = load_candidates(args.dry_run)
        if not args.dry_run:
            print(f"Pushing 'candidates' config to {args.repo_id}...")
            dd.push_to_hub(args.repo_id, config_name="candidates", token=token)
            print("  Done.")

    if not args.dry_run:
        from cnake_charmer.checkout.hf import init_registry

        print("Initializing checkout registry...")
        init_registry(args.repo_id, token=token)
        print("  data/checkouts.jsonl ready on HF.")
        print(
            "\nDone. To remove DuckDB files from git tracking:\n"
            "  git rm --cached utils/stack_data/stack_cython_full.duckdb sft_candidates.duckdb"
        )
    else:
        print("\nDry run complete — nothing pushed.")


if __name__ == "__main__":
    main()
