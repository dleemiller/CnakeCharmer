#!/usr/bin/env python3
"""
downloader.py – config-driven toolkit for The Stack / SWH datasets
=================================================================

*Import it as a library or run from CLI.*

Core object: Downloader
• Exports metadata (blob_id, src_encoding, directory_id, repo_id) → SQLite, with `content` column NULL
• Downloads missing blobs into `content`
• Helpers: explore / topdir

You can set `max_batches` in your YAML *or* via `--max-batches` on the CLI.

Dependencies: pip install polars aiohttp tqdm pyyaml
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import math
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import Any, Sequence

import aiohttp
import polars as pl
import yaml  # type: ignore
from tqdm import tqdm


def _decompress_if_needed(data: bytes) -> bytes:
    return gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data


async def _http_fetch(
    session: aiohttp.ClientSession,
    url: str,
    *,
    max_size: int | None,
    retries: int,
) -> bytes | str | None:
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status)
                if max_size is not None:
                    hdr = resp.headers.get("Content-Length")
                    if hdr and int(hdr) > max_size:
                        return "SKIPPED_LARGE_FILE"
                raw = await resp.read()
                return _decompress_if_needed(raw)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == retries - 1:
                return None
            await asyncio.sleep(1.5**attempt)
    return None


class Downloader:
    def __init__(
        self,
        *,
        sqlite_db: str | Path,
        base_url: str,
        parquet_path: str | Path | None = None,
        table: str = "files",
        batch: int = 500,
        concurrency: int = 20,
        max_size: int = 5 * 1024 * 1024,
        retries: int = 3,
        max_batches: int | None = None,
        **_: Any,
    ) -> None:
        self.parquet_path = (Path(parquet_path).expanduser().resolve() if parquet_path else None)
        self.sqlite_db = Path(sqlite_db).expanduser().resolve()
        self.base_url = base_url.rstrip("/") + "/"
        self.table = table
        self.batch = batch
        self.concurrency = concurrency
        self.max_size = max_size
        self.retries = retries
        self.max_batches = max_batches

    def run(self) -> None:
        if self.parquet_path and not self.sqlite_db.exists():
            self._export_parquet_to_sqlite()
        asyncio.run(self.download_missing())

    def _export_parquet_to_sqlite(self) -> None:
        print(f"▶ Exporting {self.parquet_path} → {self.sqlite_db} …")
        if self.sqlite_db.exists():
            self.sqlite_db.unlink()

        df = pl.read_parquet(
            self.parquet_path,
            columns=["blob_id", "src_encoding", "directory_id", "repo_id"],
        )
        total = df.height

        conn = sqlite3.connect(self.sqlite_db)
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE {self.table} (
                blob_id      TEXT PRIMARY KEY,
                src_encoding TEXT,
                directory_id TEXT,
                repo_id      TEXT,
                content      TEXT
            );
        """)

        insert_sql = (
            f"INSERT INTO {self.table} "
            "(blob_id, src_encoding, directory_id, repo_id) VALUES (?, ?, ?, ?);"
        )
        for blob_id, enc, dir_id, repo_id in tqdm(
            df.rows(), total=total, desc="export metadata", unit="row"
        ):
            cur.execute(insert_sql, (blob_id, enc, dir_id, repo_id))
        conn.commit()
        conn.close()

        print(f"✔ Exported {total:,} metadata rows (content NULL)")

    async def download_missing(self) -> None:
        db = sqlite3.connect(self.sqlite_db)
        db.row_factory = sqlite3.Row
        # ensure content column exists
        cols = [r[1] for r in db.execute(f"PRAGMA table_info({self.table})")]
        if "content" not in cols:
            db.execute(f"ALTER TABLE {self.table} ADD COLUMN content TEXT")
            db.commit()

        total_missing = db.execute(
            f"SELECT COUNT(*) FROM {self.table} WHERE content IS NULL"
        ).fetchone()[0]
        print(f"Need to fetch {total_missing:,} blobs…")

        # compute how many batches we'll actually do
        total_batches = math.ceil(total_missing / self.batch)
        if self.max_batches is not None:
            total_batches = min(total_batches, self.max_batches)

        batch_bar = tqdm(total=total_batches, desc="batches", unit="batch")
        overall  = tqdm(total=total_missing, desc="blobs",   unit="blob")
        sem = asyncio.Semaphore(self.concurrency)
        batch_num = 0
        async with aiohttp.ClientSession() as session:
            while True:
                rows = db.execute(
                    f"SELECT blob_id, src_encoding FROM {self.table} "
                    "WHERE content IS NULL LIMIT ?",
                    (self.batch,),
                ).fetchall()
                if not rows or (self.max_batches is not None and batch_num >= self.max_batches):
                    break

                batch_num += 1
                tasks = [
                    asyncio.create_task(self._worker(sem, session, blob_id, enc))
                    for blob_id, enc in rows
                ]

                # download this batch (we leave the batch progress hidden)
                for fut in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc=f"batch {batch_num}",
                    unit="blob",
                    leave=False,
                ):
                    content, blob = await fut
                    overall.update(1)
                    db.execute(
                        f"UPDATE {self.table} SET content = ? WHERE blob_id = ?",
                        (content, blob),
                    )
                db.commit()
                batch_bar.update(1)

        batch_bar.close()
        overall.close()
        db.close()
        print("✔ Downloads complete")

    async def _worker(
        self,
        sem: asyncio.Semaphore,
        session: aiohttp.ClientSession,
        blob_id: str,
        encoding: str,
    ) -> tuple[str, str]:
        async with sem:
            data = await _http_fetch(
                session,
                self.base_url + blob_id,
                max_size=self.max_size,
                retries=self.retries,
            )
            if data is None:
                return "ERROR", blob_id
            if data == "SKIPPED_LARGE_FILE":
                return "SKIPPED_LARGE_FILE", blob_id
            try:
                text = data.decode(encoding)  # type: ignore
            except (UnicodeDecodeError, AttributeError):
                try:
                    text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                except UnicodeDecodeError:
                    return "SKIPPED_BINARY_CONTENT", blob_id
            return text, blob_id


def explore_parquet(path: Path, *, rows: int = 10, lazy: bool = False) -> None:
    df = pl.scan_parquet(path) if lazy else pl.read_parquet(path)
    print("Rows × Cols:", df.shape if not lazy else "(lazy – unknown)")
    print(df.schema)
    print(df.head(rows))


def top_directory(path: Path, *, rows: int = 20, lazy: bool = False) -> None:
    lf = pl.scan_parquet(path) if lazy else pl.read_parquet(path)
    top_id = (
        lf.groupby("directory_id")
        .count()
        .sort("count", descending=True)
        .limit(1)
        .collect()[0, "directory_id"]
    )
    print("Top directory:", top_id)
    print(lf.filter(pl.col("directory_id") == top_id).limit(rows).collect())


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(path.read_text())
    if path.suffix == ".json":
        import json as _j

        return _j.loads(path.read_text())
    raise ValueError("Config must be YAML or JSON")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Downloader for The Stack / Software Heritage blobs.

Sub-commands:
  run     – export (if needed) + download
  explore – quick peek at a Parquet shard
  topdir  – show directory with most blobs
"""
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Export → download")
    runp.add_argument("--config",      type=Path, help="YAML/JSON config file")
    runp.add_argument("--max-batches", type=int,  help="stop after N download batches (overrides config)")
    runp.add_argument("--parquet",     type=Path, help="Parquet file (if no config)")
    runp.add_argument("--sqlite",      type=Path, help="SQLite DB (if no config)")
    runp.add_argument("--base-url",    help="Base URL (if no config)")
    runp.add_argument("--table",       default="files",       help="DB table name")
    runp.add_argument("--batch",       type=int, default=500, help="rows per batch")
    runp.add_argument("--concurrency", type=int, default=20,  help="HTTP concurrency")
    runp.add_argument("--max-size",    type=int, default=5*1024*1024, help="max bytes")
    runp.add_argument("--retries",     type=int, default=3,   help="retry attempts")

    xp = sub.add_parser("explore", help="Quick peek at Parquet")
    xp.add_argument("--path", type=Path, required=True)
    xp.add_argument("--rows", type=int, default=10)
    xp.add_argument("--lazy", action="store_true")

    td = sub.add_parser("topdir", help="Show directory with most blobs")
    td.add_argument("--path", type=Path, required=True)
    td.add_argument("--rows", type=int, default=20)
    td.add_argument("--lazy", action="store_true")

    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd == "explore":
        explore_parquet(args.path, rows=args.rows, lazy=args.lazy)
        return
    if args.cmd == "topdir":
        top_directory(args.path, rows=args.rows, lazy=args.lazy)
        return

    if args.config:
        cfg = load_config(args.config)
        if args.max_batches is not None:
            cfg["max_batches"] = args.max_batches
    else:
        if not (args.parquet and args.sqlite and args.base_url):
            sys.exit("error: --parquet, --sqlite and --base-url are required when no --config")
        cfg = {
            "parquet_path": str(args.parquet),
            "sqlite_db":    str(args.sqlite),
            "base_url":     args.base_url,
            "table":        args.table,
            "batch":        args.batch,
            "concurrency":  args.concurrency,
            "max_size":     args.max_size,
            "retries":      args.retries,
            "max_batches":  args.max_batches,
        }

    Downloader(**cfg).run()


if __name__ == "__main__":
    main()
