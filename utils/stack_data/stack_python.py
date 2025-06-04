#!/usr/bin/env python3
"""
stack_python.py – Swiss‑army knife for Parquet workflows
=======================================================

Sub‑commands:
* **explore**  – overview stats.
* **topdir**   – show rows from the most‑populated `directory_id`.
* **export**   – stream Parquet → SQLite via DuckDB.
* **save**     – write source files to disk by **directory_id** or **blob_id**.

Quick demo
----------
```bash
# Save every file in a directory_id
python utils/db_file_maker/stack_python.py save \
       --path ./data/train-00000-of-00006.parquet \
       --directory-id 5125127fb6bbdb25af273b
```

If you prefer to specify particular blobs:
```bash
python stack_python.py save --path data/p.parquet \
       --blob-id aff1a9263e183610f403a4d,6a7f27b...
```

Requirements
------------
```
pip install polars duckdb pytest
```
"""

from __future__ import annotations

import argparse
import functools
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import polars as pl

# ``duckdb`` powers Parquet → SQLite export
try:
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover – export command will complain
    duckdb = None  # type: ignore

###############################################################################
# Explore helpers
###############################################################################

def explore_parquet(path: Path, *, lazy: bool = False, rows: int = 10) -> None:
    """Pretty‑print a high‑signal overview of *path* using Polars."""

    df = pl.scan_parquet(path) if lazy else pl.read_parquet(path)

    # ── Shape & schema ────────────────────────────────────────────────
    print("Rows × Cols:", df.shape if not lazy else "(lazy) — unknown rows until collect()")
    print("\nSchema:")
    print(df.schema)

    # ── Head ──────────────────────────────────────────────────────────
    head = df.head(rows) if not lazy else df.limit(rows).collect()
    print(f"\nHead (first {rows} rows):")
    print(head)

    # ── Numeric describe ─────────────────────────────────────────────
    describe = df.describe() if not lazy else df.select(pl.all().describe()).collect()
    print("\nDescribe numeric columns:")
    print(describe)

    # ── Distinct counts ──────────────────────────────────────────────
    if not lazy:
        print("\nDistinct counts per column:")
        for col in df.columns:
            print(f"{col:>24}: {df[col].n_unique()}")

    # ── Null overview ────────────────────────────────────────────────
    nulls = (
        df.null_count()
        .transpose()
        .with_columns((pl.col("column_0") / (df.height if not lazy else head.height)).alias("null_fraction"))
    )
    print("\nNull counts & fractions:")
    print(nulls.sort("column_0", descending=True))

###############################################################################
# Top‑directory helper
###############################################################################

def show_top_directory(path: Path, *, lazy: bool = False, rows: int = 20) -> None:
    """Find directory_id with max blobs and print its rows."""

    if lazy:
        df_lazy = pl.scan_parquet(path)
        counts = df_lazy.groupby("directory_id").count().sort("count", descending=True).limit(1).collect()
        top_id, total = counts.row(0)
        print(f"Top directory: {top_id}  (files: {total:,})")
        target_rows = (
            df_lazy.filter(pl.col("directory_id") == top_id)
            .limit(rows)
            .collect()
        )
    else:
        df = pl.read_parquet(path)
        counts = df.groupby("directory_id").count().sort("count", descending=True)
        top_id, total = counts.row(0)
        print(f"Top directory: {top_id}  (files: {total:,})")
        target_rows = df.filter(pl.col("directory_id") == top_id).head(rows)

    print(f"\nRows from that directory (showing up to {rows}):")
    print(target_rows)

###############################################################################
# Save helper
###############################################################################

def _ensure_content_column(df: pl.DataFrame) -> str:
    """Return name of column containing source code / file bytes."""
    for candidate in ("content", "source", "code", "text"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No column named 'content', 'source', 'code', or 'text' found. Columns: "
        + ", ".join(df.columns)
    )


def _collect_target_rows(
    parquet_path: Path,
    *,
    directory_ids: Sequence[str] | None,
    blob_ids: Sequence[str] | None,
) -> pl.DataFrame:
    """Load parquet lazily and collect rows matching criteria."""

    if not directory_ids and not blob_ids:
        raise ValueError("Specify --directory-id and/or --blob-id.")

    df_lazy = pl.scan_parquet(parquet_path)

    conds: list[pl.Expr] = []
    if directory_ids:
        conds.append(pl.col("directory_id").is_in(directory_ids))
    if blob_ids:
        conds.append(pl.col("blob_id").is_in(blob_ids))

    # Combine with OR (|). Avoid previous bug with `pl.any(conds)`.
    combined = functools.reduce(lambda a, b: a | b, conds)
    return df_lazy.filter(combined).collect()


def save_files(
    parquet_path: Path,
    *,
    directory_ids: Sequence[str] | None = None,
    blob_ids: Sequence[str] | None = None,
    outdir: Path = Path("code"),
) -> None:
    """Write files to *outdir*/{directory_id}/{path}."""

    target = _collect_target_rows(parquet_path, directory_ids=directory_ids, blob_ids=blob_ids)

    if target.is_empty():
        print("No matching rows – nothing to write.")
        return

    content_col = _ensure_content_column(target)

    print(f"▶ Saving {len(target):,} file(s) to {outdir}/…")

    written = 0
    for row in target.iter_rows(named=True):
        dir_id: str = row["directory_id"]
        rel_path: str = str(row["path"]).lstrip("/")
        dest = outdir / dir_id / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        content = row[content_col]
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
            dest.write_text(text, encoding="utf-8")
        else:
            dest.write_text(str(content), encoding="utf-8")

        written += 1
    print(f"✔ Done – {written} file(s) written.")

###############################################################################
# Export helper
###############################################################################

def export_parquet_to_sqlite(
    parquet_path: Path,
    sqlite_db: Path,
    table_name: str = "files",
    *,
    overwrite: bool = False,
) -> None:
    """Stream *parquet_path* into *sqlite_db* via DuckDB."""

    if duckdb is None:
        raise ImportError("duckdb package required. Install with `pip install duckdb`.")

    if sqlite_db.exists() and not overwrite:
        raise FileExistsError(f"{sqlite_db} exists. Use --overwrite to replace.")

    print("▶ Starting Parquet → SQLite export…")

    con = duckdb.connect()
    con.execute(
        "COPY (SELECT * FROM parquet_scan(?)) TO ? (FORMAT 'sqlite', TABLE ?, OVERWRITE ?);",
        (str(parquet_path), str(sqlite_db), table_name, overwrite),
    )
    row_count = con.execute(
        "SELECT COUNT(*) FROM sqlite_scan(?, ?);", (str(sqlite_db), table_name)
    ).fetchone()[0]
    con.close()

    print(f"✔ Export complete → {sqlite_db}  (rows: {row_count:,})")

###############################################################################
# CLI glue
###############################################################################

def _csv_seq(s: str) -> list[str]:
    return [part.strip() for part in s.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Explore or transform a Parquet file containing source‑code metadata.

Commands:
  explore   – overview stats
  topdir    – directory with most blobs
  export    – parquet → sqlite
  save      – write out real files for inspection
"""
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # explore --------------------------------------------------------------
    px = sub.add_parser("explore", help="Overview of the Parquet file.")
    px.add_argument("--path", type=Path, default=Path("./data/train-00000-of-00006.parquet"))
    px.add_argument("--rows", type=int, default=10)
    px.add_argument("--lazy", action="store_true")

    # topdir ---------------------------------------------------------------
    pd_parser = sub.add_parser("topdir", help="Show rows from the busiest directory_id.")
    pd_parser.add_argument("--path", type=Path, default=Path("./data/train-00000-of-00006.parquet"))
    pd_parser.add_argument("--rows", type=int, default=20)
    pd_parser.add_argument("--lazy", action="store_true")

    # export ---------------------------------------------------------------
    pe = sub.add_parser("export", help="Convert Parquet → SQLite via DuckDB.")
    pe.add_argument("--path", type=Path, required=True)
    pe.add_argument("--sqlite-db", type=Path, required=True)
    pe.add_argument("--table", default="files")
    pe.add_argument("--overwrite", action="store_true")

    # save -----------------------------------------------------------------
    ps = sub.add_parser("save", help="Write files to disk by directory_id/blob_id.")
    ps.add_argument("--path", type=Path, required=True)
    ps.add_argument("--directory-id", type=_csv_seq)
    ps.add_argument("--blob-id", type=_csv_seq)
    ps.add_argument("--outdir", type=Path, default=Path("code"))

    return p


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _build_parser().parse_args(argv)

    if args.cmd == "explore":
        explore_parquet(args.path, lazy=args.lazy, rows=args.rows)
    elif args.cmd == "topdir":
        show_top_directory(args.path, lazy=args.lazy, rows=args.rows)
    elif args.cmd == "export":
        export_parquet_to_sqlite(
            args.path,
            args.sqlite_db,
            table_name=args.table,
            overwrite=args.overwrite,
        )
    elif args.cmd == "save":
        save_files(
            args.path,
            directory_ids=args.directory_id,
            blob_ids=args.blob_id,
            outdir=args.outdir,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()



"""
# 1. Dump every file under the busiest directory into ./code/<dir_id>/...
python utils/db_file_maker/stack_python.py save \
       --path ./data/train-00000-of-00006.parquet \
       --directory-id ca7aa979e7059467e158830b76673f5b77a0f5a3

# 2. Grab a couple of specific blobs
python stack_python.py save --path ./data/train-00000-of-00006.parquet \
       --blob-id aff1a9263e183610f403a4d,6a7f27b... \
       --outdir ./scratch/code
"""