"""
Trace reflection engine — extracts patterns from agent traces for wiki curation.

Parses structured observation text from traces to identify:
- Common compilation errors and their fixes
- Test failure patterns (output mismatches, segfaults)
- Optimization strategies (annotation improvements, speedup techniques)
- Error→fix sequences across iteration steps

Used by both the CLI script (scripts/wiki_reflect.py) and the MCP tool.
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from cnake_charmer.traces.io import load_traces
from cnake_charmer.traces.models import Trace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation parsers
# ---------------------------------------------------------------------------

# Patterns for structured observation sections
_RE_SEGFAULT = re.compile(r"exit -11|segfault", re.IGNORECASE)
_RE_TIMEOUT = re.compile(r"timed?\s*out|killed.*time", re.IGNORECASE)
_RE_COMPILE_ERROR = re.compile(r"Compilation failed")
_RE_COMPILE_OK = re.compile(r"Compilation successful")
_RE_ANNOTATION = re.compile(
    r"Annotation score:\s*([\d.]+)\s*\((\d+)\s*Python-fallback lines\s*/\s*(\d+)\s*total\)"
)
_RE_TESTS = re.compile(r"Tests:\s*(\d+)/(\d+)\s*passed")
_RE_SPEEDUP = re.compile(r"Speedup:\s*([\d.]+)x")
_RE_FAIL_MISMATCH = re.compile(r"FAIL:.*Output mismatch")
_RE_FAIL_CASE = re.compile(r"FAIL:\s*Case\s*\d+:\s*(.*)")
_RE_CYTHON_ERROR = re.compile(r"^(.*\.pyx:\d+:\d+:\s*.+)$", re.MULTILINE)
_RE_GIL_ERROR = re.compile(r"not allowed without the GIL|Coercion from Python.*without.*GIL")
_RE_CDEF_ERROR = re.compile(r"Cdef statement not allowed here|cdef.*not allowed")
_RE_NOGIL_ERROR = re.compile(r"not allowed in nogil|Operation not allowed without gil")


class StepAnalysis:
    """Parsed analysis of a single trace step's observation."""

    def __init__(self, observation: str):
        self.raw = observation
        self.compiled = bool(_RE_COMPILE_OK.search(observation))
        self.compile_failed = bool(_RE_COMPILE_ERROR.search(observation))
        self.segfault = bool(_RE_SEGFAULT.search(observation))
        self.timeout = bool(_RE_TIMEOUT.search(observation))

        # Compilation error details
        self.compile_errors: list[str] = _RE_CYTHON_ERROR.findall(observation)
        self.gil_error = bool(_RE_GIL_ERROR.search(observation))
        self.cdef_error = bool(_RE_CDEF_ERROR.search(observation))
        self.nogil_error = bool(_RE_NOGIL_ERROR.search(observation))

        # Annotation
        ann_match = _RE_ANNOTATION.search(observation)
        self.annotation_score = float(ann_match.group(1)) if ann_match else None
        self.yellow_lines = int(ann_match.group(2)) if ann_match else None
        self.total_lines = int(ann_match.group(3)) if ann_match else None

        # Tests
        test_match = _RE_TESTS.search(observation)
        self.tests_passed = int(test_match.group(1)) if test_match else None
        self.tests_total = int(test_match.group(2)) if test_match else None
        self.test_mismatches = len(_RE_FAIL_MISMATCH.findall(observation))
        self.fail_reasons = _RE_FAIL_CASE.findall(observation)

        # Speedup
        sp_match = _RE_SPEEDUP.search(observation)
        self.speedup = float(sp_match.group(1)) if sp_match else None

    @property
    def error_type(self) -> str | None:
        """Classify the primary error type for this step."""
        if self.segfault:
            return "segfault"
        if self.compile_failed:
            if self.gil_error:
                return "gil_violation"
            if self.cdef_error:
                return "cdef_placement"
            if self.nogil_error:
                return "nogil_violation"
            return "compilation_error"
        if self.timeout:
            return "timeout"
        if (
            self.tests_passed is not None
            and self.tests_total is not None
            and self.tests_passed < self.tests_total
        ):
            return "test_failure"
        return None


def analyze_trace(trace: Trace) -> dict:
    """Analyze a single trace for patterns."""
    steps = [StepAnalysis(s.observation) for s in trace.steps if s.tool_name != "finish"]

    # Track error→fix sequences
    error_fix_pairs = []
    for i in range(len(steps) - 1):
        cur, nxt = steps[i], steps[i + 1]
        err = cur.error_type
        if err and nxt.error_type != err:
            error_fix_pairs.append(
                {
                    "error_type": err,
                    "fixed_in_next": nxt.error_type is None,
                    "step": i,
                }
            )

    # Annotation progression
    annotations = [
        {"step": i, "score": s.annotation_score}
        for i, s in enumerate(steps)
        if s.annotation_score is not None
    ]

    # Speedup progression
    speedups = [
        {"step": i, "speedup": s.speedup} for i, s in enumerate(steps) if s.speedup is not None
    ]

    # All error types encountered
    errors = [s.error_type for s in steps if s.error_type]

    # Compile error messages
    compile_errors = []
    for s in steps:
        compile_errors.extend(s.compile_errors)

    return {
        "problem_id": trace.problem_id,
        "category": trace.category,
        "model": trace.model,
        "reward": trace.reward,
        "num_steps": len(steps),
        "errors": errors,
        "error_fix_pairs": error_fix_pairs,
        "annotations": annotations,
        "speedups": speedups,
        "compile_errors": compile_errors,
        "final_annotation": annotations[-1]["score"] if annotations else None,
        "final_speedup": speedups[-1]["speedup"] if speedups else None,
    }


# ---------------------------------------------------------------------------
# Wiki page mapping — which patterns belong on which page
# ---------------------------------------------------------------------------

_ERROR_TO_PAGE = {
    "segfault": "pitfalls",
    "gil_violation": "parallelism",
    "cdef_placement": "pitfalls",
    "nogil_violation": "parallelism",
    "compilation_error": "pitfalls",
    "test_failure": "pitfalls",
    "timeout": "optimization",
}

_COMPILE_ERROR_TO_PAGE = {
    "GIL": "parallelism",
    "nogil": "parallelism",
    "cdef": "typing",
    "memoryview": "memoryviews",
    "cimport": "c-interop",
    "cppclass": "cpp-interop",
    "struct": "c-interop",
    "malloc": "memory-management",
    "free": "memory-management",
    "enum": "enums-tuples",
    "except": "error-handling",
    "prange": "parallelism",
}


def _classify_compile_error(error_msg: str) -> str:
    """Map a compilation error message to a wiki page."""
    error_lower = error_msg.lower()
    for keyword, page in _COMPILE_ERROR_TO_PAGE.items():
        if keyword.lower() in error_lower:
            return page
    return "pitfalls"


# ---------------------------------------------------------------------------
# Aggregate reflection
# ---------------------------------------------------------------------------


def reflect_on_traces(
    traces_path: str | Path,
    category: str | None = None,
    problem_id: str | None = None,
    min_reward: float = 0.0,
    max_traces: int = 500,
    since: str | None = None,
) -> dict:
    """Analyze traces and produce findings grouped by wiki page.

    Args:
        traces_path: Path to master_traces.jsonl.
        category: Filter by problem category.
        problem_id: Filter by specific problem.
        min_reward: Minimum reward threshold.
        max_traces: Max traces to process.
        since: ISO date string — only include traces after this date.

    Returns:
        Dict with summary stats and findings grouped by wiki page.
    """
    traces = load_traces([traces_path])
    logger.info(f"Loaded {len(traces)} traces")

    # Filter
    since_dt = datetime.fromisoformat(since) if since else None
    filtered = []
    for t in traces:
        if category and t.category != category:
            continue
        if problem_id and t.problem_id != problem_id:
            continue
        if t.reward < min_reward:
            continue
        if since_dt and t.timestamp and t.timestamp < since_dt:
            continue
        filtered.append(t)
        if len(filtered) >= max_traces:
            break

    logger.info(f"Analyzing {len(filtered)} traces after filtering")

    if not filtered:
        return {"summary": {"total_traces": 0}, "findings": {}}

    # Analyze all traces
    analyses = [analyze_trace(t) for t in filtered]

    # Aggregate stats
    error_counts = Counter()
    error_by_category = defaultdict(Counter)
    compile_error_msgs = Counter()
    page_findings = defaultdict(
        lambda: {
            "error_patterns": Counter(),
            "compile_errors": Counter(),
            "fix_success_rate": {"fixed": 0, "total": 0},
            "example_traces": [],
        }
    )

    speedups = []
    annotations = []
    steps_to_success = []

    for a in analyses:
        # Error counts
        for err in a["errors"]:
            error_counts[err] += 1
            error_by_category[a["category"]][err] += 1

            page = _ERROR_TO_PAGE.get(err, "pitfalls")
            page_findings[page]["error_patterns"][err] += 1

        # Compile error messages → pages
        for msg in a["compile_errors"]:
            compile_error_msgs[msg] += 1
            page = _classify_compile_error(msg)
            # Truncate long messages for readability
            short_msg = msg[:120] if len(msg) > 120 else msg
            page_findings[page]["compile_errors"][short_msg] += 1

        # Error→fix tracking
        for pair in a["error_fix_pairs"]:
            page = _ERROR_TO_PAGE.get(pair["error_type"], "pitfalls")
            page_findings[page]["fix_success_rate"]["total"] += 1
            if pair["fixed_in_next"]:
                page_findings[page]["fix_success_rate"]["fixed"] += 1

        # Annotation/speedup stats
        if a["final_annotation"] is not None:
            annotations.append(a["final_annotation"])
        if a["final_speedup"] is not None:
            speedups.append(a["final_speedup"])
        steps_to_success.append(a["num_steps"])

        # Keep a few example traces for high-error pages
        if a["errors"]:
            page = _ERROR_TO_PAGE.get(a["errors"][0], "pitfalls")
            examples = page_findings[page]["example_traces"]
            if len(examples) < 3:
                examples.append(
                    {
                        "problem_id": a["problem_id"],
                        "errors": a["errors"],
                        "reward": round(a["reward"], 3),
                    }
                )

    # Build summary
    summary = {
        "total_traces": len(filtered),
        "traces_with_errors": sum(1 for a in analyses if a["errors"]),
        "error_counts": dict(error_counts.most_common(20)),
        "avg_steps": round(sum(steps_to_success) / len(steps_to_success), 1),
        "avg_speedup": round(sum(speedups) / len(speedups), 1) if speedups else None,
        "avg_annotation": round(sum(annotations) / len(annotations), 3) if annotations else None,
        "top_compile_errors": dict(compile_error_msgs.most_common(10)),
    }

    if error_by_category:
        summary["errors_by_category"] = {
            cat: dict(counts.most_common(5)) for cat, counts in sorted(error_by_category.items())
        }

    # Serialize page findings
    findings = {}
    for page, data in sorted(page_findings.items()):
        fix = data["fix_success_rate"]
        findings[page] = {
            "error_patterns": dict(data["error_patterns"].most_common(10)),
            "compile_errors": dict(data["compile_errors"].most_common(10)),
            "fix_success_rate": (
                round(fix["fixed"] / fix["total"], 2) if fix["total"] > 0 else None
            ),
            "fix_attempts": fix["total"],
            "example_traces": data["example_traces"],
        }

    return {"summary": summary, "findings": findings}
