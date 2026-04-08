"""
Trace analysis utilities for LLM-powered wiki reflection.

Bridges the gap between regex-only reflection (which only reads observation text)
and actual code changes (which live in tool_args["code"]).
"""

import difflib
from pathlib import Path

from cnake_charmer.traces.models import Trace
from cnake_charmer.wiki.reflect import _ERROR_TO_PAGE, StepAnalysis

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CY_DIR = _PROJECT_ROOT / "cnake_data" / "cy"
_DOCS_DIR = _PROJECT_ROOT / ".sources" / "cython-docs" / "docs" / "src"

# Map wiki pages to categories for loading reference implementations
_PAGE_TO_CATEGORIES = {
    "typing": ["numerical", "algorithms"],
    "memoryviews": ["numerical", "dsp"],
    "numpy-interop": ["numerical", "statistics"],
    "parallelism": ["simulation", "numerical"],
    "optimization": ["algorithms", "sorting"],
    "compiler-directives": ["numerical", "algorithms"],
    "pitfalls": ["algorithms", "numerical"],
    "error-handling": ["algorithms", "graph"],
    "memory-management": ["algorithms", "graph"],
    "c-interop": ["algorithms", "cryptography"],
    "cpp-interop": ["algorithms", "graph"],
    "extension-types": ["graph", "geometry"],
    "enums-tuples": ["algorithms", "geometry"],
}


def extract_code_diffs(trace: Trace) -> list[dict]:
    """Extract code diffs between consecutive evaluate_cython steps.

    Returns list of dicts with: step index, error before/after, code diff summary.
    Only includes steps where `tool_name == "evaluate_cython"` and code changed.
    """
    eval_steps = []
    for i, step in enumerate(trace.steps):
        if step.tool_name == "evaluate_cython" and "code" in step.tool_args:
            analysis = StepAnalysis(step.observation)
            eval_steps.append((i, step.tool_args["code"], analysis))

    diffs = []
    for idx in range(len(eval_steps) - 1):
        step_i, code_before, analysis_before = eval_steps[idx]
        step_j, code_after, analysis_after = eval_steps[idx + 1]

        diff_lines = list(
            difflib.unified_diff(
                code_before.splitlines(),
                code_after.splitlines(),
                lineterm="",
                n=2,
            )
        )

        if not diff_lines:
            continue

        diffs.append(
            {
                "step": step_i,
                "error_before": analysis_before.error_type,
                "error_after": analysis_after.error_type,
                "fixed": analysis_before.error_type and not analysis_after.error_type,
                "diff": "\n".join(diff_lines[:50]),  # cap for context
            }
        )

    return diffs


def select_example_traces(traces: list[Trace], page: str, max_examples: int = 5) -> list[Trace]:
    """Select diverse, informative traces for a wiki page.

    Prioritizes traces with:
    - Errors mapped to this page
    - Mix of successful fixes and failures
    - Diverse error types
    """
    scored = []
    for trace in traces:
        score = 0
        has_page_error = False

        for step in trace.steps:
            if step.tool_name != "evaluate_cython":
                continue
            analysis = StepAnalysis(step.observation)
            err = analysis.error_type
            if err and _ERROR_TO_PAGE.get(err, "pitfalls") == page:
                has_page_error = True
                score += 2

        if not has_page_error:
            continue

        # Prefer traces with code diffs (shows actual fixes)
        diffs = extract_code_diffs(trace)
        if any(d["fixed"] for d in diffs):
            score += 3  # successfully fixed
        if diffs:
            score += 1

        # Prefer higher reward (more complete solutions)
        score += trace.reward * 2

        scored.append((score, trace))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by problem_id for diversity
    seen_problems = set()
    selected = []
    for _, trace in scored:
        if trace.problem_id in seen_problems:
            continue
        seen_problems.add(trace.problem_id)
        selected.append(trace)
        if len(selected) >= max_examples:
            break

    return selected


def format_trace_for_llm(trace: Trace, max_code_lines: int = 40) -> str:
    """Format a trace as concise text for LLM consumption.

    Shows problem_id, reward, then per-step: thought + code diff + observation summary.
    """
    lines = [
        f"## Trace: {trace.problem_id} (reward={trace.reward:.3f}, model={trace.model})",
        "",
    ]

    diffs = extract_code_diffs(trace)
    diff_by_step = {d["step"]: d for d in diffs}

    for i, step in enumerate(trace.steps):
        if step.tool_name != "evaluate_cython":
            continue

        analysis = StepAnalysis(step.observation)
        lines.append(f"### Step {i}")

        if step.thought:
            lines.append(f"**Thought:** {step.thought[:200]}")

        if i in diff_by_step:
            d = diff_by_step[i]
            diff_lines = d["diff"].splitlines()
            if len(diff_lines) > max_code_lines:
                diff_lines = diff_lines[:max_code_lines] + [
                    f"... ({len(diff_lines) - max_code_lines} more lines)"
                ]
            lines.append("```diff")
            lines.extend(diff_lines)
            lines.append("```")

        # Observation summary
        obs_parts = []
        if analysis.compile_failed:
            obs_parts.append(f"compile_error: {analysis.error_type}")
            if analysis.compile_errors:
                obs_parts.append(f"  {analysis.compile_errors[0][:100]}")
        elif analysis.compiled:
            obs_parts.append("compiled: OK")
        if analysis.tests_passed is not None:
            obs_parts.append(f"tests: {analysis.tests_passed}/{analysis.tests_total}")
        if analysis.speedup is not None:
            obs_parts.append(f"speedup: {analysis.speedup:.1f}x")
        if analysis.annotation_score is not None:
            obs_parts.append(f"annotation: {analysis.annotation_score:.3f}")

        if obs_parts:
            lines.append("**Result:** " + " | ".join(obs_parts))
        lines.append("")

    return "\n".join(lines)


def load_reference_code(page: str, max_files: int = 3) -> str:
    """Load relevant cy/ implementations for a wiki page.

    Returns concatenated source of up to max_files .pyx files from
    categories mapped to this page.
    """
    categories = _PAGE_TO_CATEGORIES.get(page, ["numerical"])
    files_found = []

    for cat in categories:
        cat_dir = _CY_DIR / cat
        if not cat_dir.exists():
            continue
        for pyx in sorted(cat_dir.glob("*.pyx")):
            files_found.append(pyx)
            if len(files_found) >= max_files:
                break
        if len(files_found) >= max_files:
            break

    if not files_found:
        return "No reference implementations found."

    parts = []
    for pyx in files_found:
        content = pyx.read_text()
        # Truncate very long files
        lines = content.splitlines()
        if len(lines) > 80:
            lines = lines[:80] + [f"# ... ({len(lines) - 80} more lines)"]
        parts.append(f"### {pyx.relative_to(_PROJECT_ROOT)}\n```cython\n{chr(10).join(lines)}\n```")

    return "\n\n".join(parts)


def load_cython_docs(topic: str, max_chars: int = 8000) -> str:
    """Search .sources/cython-docs/docs/src/ for relevant official docs.

    Simple keyword search over .rst files, returns best matching content.
    """
    if not _DOCS_DIR.exists():
        return "Cython docs not available. Run `make wiki-setup` to clone them."

    terms = [t.lower() for t in topic.split() if t]
    if not terms:
        return "Empty topic."

    scored = []
    for rst in _DOCS_DIR.rglob("*.rst"):
        try:
            content = rst.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        content_lower = content.lower()
        score = sum(content_lower.count(t) for t in terms)
        if score > 0:
            scored.append((score, rst, content))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return f"No docs found for '{topic}'."

    # Return top result, truncated
    _, path, content = scored[0]
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... (truncated, {len(content)} total chars)"

    return f"## {path.relative_to(_DOCS_DIR)}\n\n{content}"
