"""
Heuristic difficulty classifier for Cython problems.

Scores problems based on features present in the ground-truth .pyx,
then buckets into easy/medium/hard. Used for curriculum learning
during SFT and RL training.
"""

import re
from pathlib import Path

# Feature detectors: (pattern, score, description)
_FEATURES = [
    # Manual memory management
    (r"\bmalloc\b|\bcalloc\b|\brealloc\b|\bfree\b", 2, "manual_memory"),
    # Extension types
    (r"\bcdef\s+class\b", 1, "cdef_class"),
    # Parallel / nogil
    (r"\bprange\b|\bwith\s+nogil\b", 1, "parallelism"),
    # Typed memoryviews
    (r"\[:\s*,|::\s*1\s*\]|\bmemoryview\b", 1, "memoryviews"),
    # Structs and unions
    (r"\bcdef\s+struct\b|\bcdef\s+union\b", 1, "structs"),
    # C++ interop
    (r"\blibcpp\b|\bcppclass\b|\bexcept\s*\+", 1, "cpp_interop"),
    # Fused types
    (r"\bfused\b", 1, "fused_types"),
    # Buffer protocol
    (r"__getbuffer__|__releasebuffer__", 1, "buffer_protocol"),
    # C string operations
    (r"\bmemcpy\b|\bmemset\b|\bmemcmp\b", 1, "c_string_ops"),
    # Pointer arithmetic
    (r"<\s*\w+\s*\*\s*>|&\w+\[", 1, "pointer_arithmetic"),
]

# Category-level difficulty adjustments
_CATEGORY_BONUS = {
    "nn_ops": 2,
    "diff_equations": 1,
    "compression": 1,
    "cryptography": 1,
    "image_processing": 1,
    "dsp": 1,
    "leetcode": -1,
}

# Thresholds for bucketing (tuned for ~40/35/25 split on 523 problems)
_EASY_MAX = 2
_MEDIUM_MAX = 4


def classify_difficulty(cython_code: str, category: str = "") -> str:
    """Classify a problem's difficulty based on its Cython ground truth.

    Args:
        cython_code: The ground-truth .pyx source code.
        category: The problem category (e.g., "nn_ops", "algorithms").

    Returns:
        "easy", "medium", or "hard"
    """
    if not cython_code.strip():
        return "easy"

    score = _compute_score(cython_code, category)

    if score <= _EASY_MAX:
        return "easy"
    elif score <= _MEDIUM_MAX:
        return "medium"
    return "hard"


def _compute_score(cython_code: str, category: str) -> int:
    """Compute raw difficulty score (0-15+)."""
    score = 0

    # LOC-based score
    lines = [ln for ln in cython_code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) > 60:
        score += 2
    elif len(lines) > 35:
        score += 1

    # Feature detection
    for pattern, points, _name in _FEATURES:
        if re.search(pattern, cython_code):
            score += points

    # Category bonus
    score += _CATEGORY_BONUS.get(category, 0)

    return score


def classify_all(problems_dir: Path | None = None) -> dict[str, str]:
    """Classify all problems by reading .pyx files from the repo.

    Returns:
        Dict mapping problem_id to difficulty level.
    """
    if problems_dir is None:
        problems_dir = Path(__file__).parent.parent / "cy"

    results = {}
    for pyx_path in sorted(problems_dir.rglob("*.pyx")):
        if pyx_path.name.startswith("__"):
            continue
        rel = pyx_path.relative_to(problems_dir)
        category = str(rel.parent) if rel.parent != Path(".") else "general"
        problem_id = f"{category}/{pyx_path.stem}"
        code = pyx_path.read_text()
        results[problem_id] = classify_difficulty(code, category)

    return results


def difficulty_summary(classifications: dict[str, str]) -> dict[str, int]:
    """Return counts per difficulty level."""
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for level in classifications.values():
        counts[level] = counts.get(level, 0) + 1
    return counts
