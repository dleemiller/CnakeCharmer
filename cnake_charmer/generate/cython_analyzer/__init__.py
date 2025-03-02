"""
Cython code analyzer package for assessing performance optimization.
"""

from .analyzer import CythonAnalyzer
from .parsing.static_parser import is_cython_code
from .reporting import (
    get_optimization_report,
    get_optimization_hints,
    get_analysis_explanation,
    metrics_to_analysis_result,
    get_hint_message,
)
from .reward import cython_analyzer_score
from .common import CythonAnalysisResult, OptimizationScoreComponents, OptimizationHint

__all__ = [
    "CythonAnalyzer",
    "is_cython_code",
    "get_optimization_report",
    "get_optimization_hints",
    "get_analysis_explanation",
    "metrics_to_analysis_result",
    "get_hint_message",
    "cython_analyzer_score",
    "CythonAnalysisResult",
    "OptimizationScoreComponents",
    "OptimizationHint",
]
