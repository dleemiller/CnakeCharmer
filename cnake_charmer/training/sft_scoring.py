"""
SFT trace scoring for dataset selection.

Ranks traces by speedup tier first, then tiebreaks within tier
by annotation, efficiency, conciseness, and compactness.

Used for: best-per-model-per-problem selection, GRPO reward baseline.

Accepts both v1 dicts and v2 Trace objects — Trace objects are converted
to v1 dicts internally so all parsing logic stays consistent.
"""

import math
import re

from cnake_charmer.traces.models import Trace


def _to_v1_dict(trace) -> dict:
    """Normalize input to v1 dict for scoring. Accepts Trace or dict."""
    if isinstance(trace, Trace):
        return trace.to_v1_dict()
    return trace


def sft_score(
    correctness: float,
    speedup: float,
    annotation: float,
    lint: float = 1.0,
    memory: float = 1.0,
    num_iters: int = 5,
    thought_tokens: int = 2000,
    code_tokens: int = 1500,
) -> float:
    """Score a trace for SFT selection.

    Quality (90%): speedup dominates, correctness and annotation support.
    Tiebreaker (10%): efficiency, conciseness, compactness — only matters
    within the same speedup tier.

    Returns:
        Score from 0.0 to ~1.0. Higher is better.
    """
    # Primary: solution quality
    perf = min(math.log2(speedup) / math.log2(100), 1.0) if speedup > 1 else 0
    quality = 0.25 * correctness + 0.40 * perf + 0.15 * annotation + 0.05 * lint + 0.05 * memory

    # Secondary: efficiency tiebreaker (never overrides quality)
    # Count only evaluate_cython calls, not finish — calling finish is good behavior
    efficiency = max(0, 1.0 - (num_iters - 1) / 4)
    conciseness = max(0, 1.0 - thought_tokens / 5000)
    compactness = max(0, 1.0 - code_tokens / 3000)
    tiebreaker = 0.05 * efficiency + 0.025 * conciseness + 0.025 * compactness

    return quality + tiebreaker


# Hard filters — trace must pass ALL of these
def passes_hard_filters(trace) -> bool:
    """Check if a trace passes all hard filters for SFT inclusion."""
    trace = _to_v1_dict(trace)
    # Must have valid tool calls (no None)
    tools = trace.get("tools_used", [])
    if None in tools or "None" in [str(t) for t in tools]:
        return False

    # Must have called evaluate_cython
    if "evaluate_cython" not in tools:
        return False

    # Must have produced code
    if not trace.get("cython_code", "").strip():
        return False

    # Must have nonzero reward (compiled + something worked)
    return (trace.get("reward") or 0) != 0


def parse_trace_metrics(trace) -> dict:
    """Extract scoring metrics from a trace record."""
    trace = _to_v1_dict(trace)
    traj = trace.get("trajectory", {})

    # Find last observation with results
    speedup = 0.0
    annotation = 0.0
    tests_passed = 0
    tests_total = 0

    idx = 0
    while f"observation_{idx}" in traj:
        obs = traj[f"observation_{idx}"] or ""
        m = re.search(r"Speedup:\s*([\d.]+)x", obs)
        if m:
            speedup = float(m.group(1))
        m = re.search(r"Annotation score:\s*([\d.]+)", obs)
        if m:
            annotation = float(m.group(1))
        m = re.search(r"Tests:\s*(\d+)/(\d+)", obs)
        if m:
            tests_passed = int(m.group(1))
            tests_total = int(m.group(2))
        idx += 1

    # Count evaluate calls (not finish — finish is good behavior, not a cost)
    eval_calls = 0
    called_finish = False
    thought_tokens = 0
    idx = 0
    while f"tool_name_{idx}" in traj:
        tool = traj.get(f"tool_name_{idx}")
        if tool == "evaluate_cython":
            eval_calls += 1
        elif tool == "finish":
            called_finish = True
        thought_tokens += len(traj.get(f"thought_{idx}", "") or "")
        idx += 1

    code_tokens = len(trace.get("cython_code", "") or "")
    correctness = tests_passed / tests_total if tests_total > 0 else 0.0

    return {
        "correctness": correctness,
        "speedup": speedup,
        "annotation": annotation,
        "num_iters": eval_calls,  # only count evaluate calls
        "called_finish": called_finish,
        "thought_tokens": thought_tokens,
        "code_tokens": code_tokens,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
    }


def score_trace(trace) -> float:
    """Score a trace for SFT selection. Returns 0.0 if hard filters fail."""
    if not passes_hard_filters(trace):
        return 0.0

    m = parse_trace_metrics(trace)

    # Hard: must pass all tests
    if m["correctness"] < 1.0:
        return 0.0

    return sft_score(
        correctness=m["correctness"],
        speedup=m["speedup"],
        annotation=m["annotation"],
        num_iters=m["num_iters"],
        thought_tokens=m["thought_tokens"],
        code_tokens=m["code_tokens"],
    )
