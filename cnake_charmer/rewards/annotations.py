"""
Annotation quality reward: based on Cython HTML annotation analysis.

Rewards code that minimizes Python-level operations and maximizes
pure C execution, as measured by Cython's annotation output.
"""

from cnake_charmer.validate.annotations import parse_annotations
from cnake_charmer.validate.compiler import cleanup_build, compile_cython


def annotation_reward(cython_code: str, **kwargs) -> float:
    """
    Return annotation score (0.0 to 1.0).

    Higher score = more C-level operations, fewer Python fallbacks.
    """
    result = compile_cython(cython_code, annotate=True, keep_build=True)

    if not result.success or not result.html_path:
        cleanup_build(result)
        return 0.0

    ann = parse_annotations(html_path=result.html_path)
    cleanup_build(result)

    if not ann.success:
        return 0.0

    return ann.score
