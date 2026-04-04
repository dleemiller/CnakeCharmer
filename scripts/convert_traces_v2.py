"""
Convert v1 flat-key traces to v2 structured Pydantic format.

Reads each JSONL file, converts traces via Trace.from_v1_dict(),
writes to a parallel output file. Original files are preserved.

Usage:
    uv run --no-sync python scripts/convert_traces_v2.py
    uv run --no-sync python scripts/convert_traces_v2.py --input data/traces/master_thinking.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.traces.models import Trace

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACES_DIR = Path("data/traces")


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Convert a single JSONL file from v1 to v2 format.

    Returns (converted_count, error_count).
    """
    converted = 0
    errors = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if raw.get("version") == "2.0":
                    # Already v2, pass through
                    fout.write(line + "\n")
                    converted += 1
                    continue
                trace = Trace.from_v1_dict(raw)
                fout.write(trace.model_dump_json() + "\n")
                converted += 1
            except Exception as e:
                logger.warning(f"  Line {line_num}: {e}")
                errors += 1
    return converted, errors


def main():
    parser = argparse.ArgumentParser(description="Convert v1 traces to v2 Pydantic format")
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=None,
        help="Input JSONL files (default: all data/traces/*.jsonl)",
    )
    parser.add_argument(
        "--suffix",
        default="_v2",
        help="Suffix for output files (default: _v2, e.g. master_thinking_v2.jsonl)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files (creates .v1_backup first)",
    )
    args = parser.parse_args()

    if args.input:
        input_files = [Path(p) for p in args.input]
    else:
        input_files = sorted(TRACES_DIR.glob("*.jsonl"))

    if not input_files:
        logger.error("No input files found")
        return

    total_converted = 0
    total_errors = 0

    for path in input_files:
        if args.in_place:
            backup = path.with_suffix(".v1_backup.jsonl")
            path.rename(backup)
            output_path = path
            source_path = backup
        else:
            stem = path.stem
            output_path = path.parent / f"{stem}{args.suffix}.jsonl"
            source_path = path

        logger.info(f"Converting {source_path.name} → {output_path.name}")
        converted, errors = convert_file(source_path, output_path)
        total_converted += converted
        total_errors += errors
        logger.info(f"  {converted} traces converted, {errors} errors")

    logger.info(f"Done: {total_converted} total converted, {total_errors} total errors")


if __name__ == "__main__":
    main()
