"""
Common data structures for Cython analysis results.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OptimizationScoreComponents:
    """Components that make up the Cython optimization score."""

    c_ratio_score: float = 0.0
    python_interaction_score: float = 0.0
    gil_operations_score: float = 0.0
    vectorizable_loops_score: float = 0.0
    feature_score: float = 0.0


@dataclass
class OptimizationHint:
    """Hint for a specific line of code to improve optimization."""

    line_number: int
    category: str
    message: str


@dataclass
class CythonAnalysisResult:
    """Complete result of Cython code analysis for reward purposes."""

    # Overall metrics
    optimization_score: float = 0.0
    c_ratio: float = 0.0
    python_ratio: float = 0.0
    feature_density: float = 0.0

    # Component scores
    component_scores: OptimizationScoreComponents = field(
        default_factory=OptimizationScoreComponents
    )

    # Feature counts
    cdef_vars: int = 0
    cdef_funcs: int = 0
    cpdef_funcs: int = 0
    memoryviews: int = 0
    typed_args: int = 0
    nogil: int = 0
    prange: int = 0
    directives: int = 0

    # Line-specific information
    optimization_hints: List[OptimizationHint] = field(default_factory=list)

    # Errors, if any
    error: Optional[str] = None
    success: bool = True
