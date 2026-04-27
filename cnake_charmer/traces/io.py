"""
Trace I/O for mixed trace formats.

`load_traces()` accepts both:
- v2 structured traces (current): `steps`, `final_code`, ...
- v1 flat traces (legacy): `trajectory`, `cython_code`, ...
"""

import json
import logging
from pathlib import Path

from cnake_charmer.traces.models import Trace

logger = logging.getLogger(__name__)


def load_traces(paths: list[str | Path]) -> list[Trace]:
    """Load traces from JSONL files (v1 + v2 compatible)."""
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
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    # v2 (current)
                    if "steps" in raw or str(raw.get("version", "")).startswith("2"):
                        traces.append(Trace.model_validate(raw))
                    # v1 (legacy flat trajectory)
                    elif "trajectory" in raw or str(raw.get("version", "")).startswith("1"):
                        traces.append(Trace.from_v1_dict(raw))
                    # Fallback: try v2 first, then v1 conversion
                    else:
                        try:
                            traces.append(Trace.model_validate(raw))
                        except Exception:
                            traces.append(Trace.from_v1_dict(raw))
                    count += 1
                except Exception as e:
                    logger.debug(f"Skipping malformed trace in {path.name}: {e}")
        logger.info(f"  {path.name}: {count} traces")
    return traces


def save_traces(traces: list[Trace], path: str | Path) -> int:
    """Save traces to JSONL in v2 format. Returns count written."""
    path = Path(path)
    count = 0
    with open(path, "w") as f:
        for trace in traces:
            f.write(trace.model_dump_json() + "\n")
            count += 1
    return count


def append_trace(trace: Trace, path: str | Path) -> None:
    """Append a single trace to a JSONL file."""
    with open(path, "a") as f:
        f.write(trace.model_dump_json() + "\n")
