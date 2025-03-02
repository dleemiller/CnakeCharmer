"""
Reporting utilities for Cython code analysis.

These functions provide human-readable reports and optimization hints
based on Cython code analysis results.
"""

from typing import Dict, Any, List, Union, Optional, Mapping
from dataclasses import asdict

from .common import CythonAnalysisResult, OptimizationHint, OptimizationScoreComponents


def get_optimization_hints(
    analysis: Union[Dict[str, Any], CythonAnalysisResult],
) -> Dict[int, str]:
    """
    Generate optimization hints based on the analysis.

    Args:
        analysis: Analysis results from CythonAnalyzer or a CythonAnalysisResult

    Returns:
        dict: Line number -> optimization hint mappings
    """
    hints = {}

    # Handle different input types
    if isinstance(analysis, CythonAnalysisResult):
        # If we received a CythonAnalysisResult, extract hints directly
        for hint in analysis.optimization_hints:
            hints[hint.line_number] = hint.message
        return hints

    # Legacy dictionary-based processing
    line_categories = analysis.get("line_categories", {})

    for line_num, category in line_categories.items():
        if category == "python_interaction":
            hints[line_num] = (
                "Consider using typed variables to avoid Python object interaction"
            )
        elif category == "gil_acquisition":
            hints[line_num] = (
                "This operation requires the GIL. Consider using 'nogil' where possible"
            )
        elif category == "vectorizable_loop":
            hints[line_num] = (
                "This loop could be vectorized using 'prange' from cython.parallel"
            )
        elif category == "unoptimized_math":
            hints[line_num] = (
                "Use typed variables for math operations to enable C-level performance"
            )

    return hints


def get_hint_message(category: str) -> str:
    """
    Get a hint message based on the category of optimization issue.

    Args:
        category: The category of the optimization issue

    Returns:
        str: A hint message suggesting how to improve the code
    """
    hint_messages = {
        "python_interaction": "Consider using typed variables to avoid Python object interaction",
        "gil_acquisition": "This operation requires the GIL. Consider using 'nogil' where possible",
        "vectorizable_loop": "This loop could be vectorized using 'prange' from cython.parallel",
        "unoptimized_math": "Use typed variables for math operations to enable C-level performance",
        "c_operation": "This line is already optimized at the C level",
    }

    return hint_messages.get(category, f"Unknown optimization category: {category}")


def get_optimization_report(
    analysis: Union[Dict[str, Any], CythonAnalysisResult],
) -> str:
    """
    Generate a detailed report of the optimization metrics.

    Args:
        analysis: Analysis results from CythonAnalyzer or a CythonAnalysisResult

    Returns:
        str: Detailed optimization report
    """
    # Handle CythonAnalysisResult input
    if isinstance(analysis, CythonAnalysisResult):
        return get_analysis_explanation(analysis)

    # Legacy dictionary-based processing
    if "error" in analysis:
        return f"Analysis failed: {analysis['error']}"

    report = ["Cython Optimization Analysis Report"]
    report.append("=" * 40)
    report.append("")

    # Basic metrics
    report.append(f"Total lines: {analysis.get('total_lines', 0)}")
    report.append(
        f"C code lines: {analysis.get('c_lines', 0)} ({analysis.get('c_ratio', 0):.2%})"
    )
    report.append(
        f"Python interaction lines: {analysis.get('python_lines', 0)} ({analysis.get('python_ratio', 0):.2%})"
    )
    report.append(f"GIL operations: {analysis.get('gil_operations', 0)}")
    report.append(f"Vectorizable loops: {analysis.get('vectorizable_loops', 0)}")
    report.append(f"Unoptimized math operations: {analysis.get('unoptimized_math', 0)}")
    report.append("")

    # Cython features
    code_features = analysis.get("code_features", {})
    if code_features:
        report.append("Cython Features:")
        report.append(
            f"- C variable declarations (cdef): {code_features.get('cdef_vars', 0)}"
        )
        report.append(f"- C functions (cdef): {code_features.get('cdef_funcs', 0)}")
        report.append(
            f"- Python-accessible C functions (cpdef): {code_features.get('cpdef_funcs', 0)}"
        )
        report.append(f"- Memory views: {code_features.get('memoryviews', 0)}")
        report.append(
            f"- Type-annotated arguments: {code_features.get('typed_args', 0)}"
        )
        report.append(f"- No-GIL sections: {code_features.get('nogil', 0)}")
        report.append(f"- Parallel loops (prange): {code_features.get('prange', 0)}")
        report.append(f"- Cython directives: {code_features.get('directives', 0)}")
        report.append("")

    # Component scores
    component_scores = analysis.get("component_scores", {})
    if component_scores:
        report.append("Optimization Score Components:")
        report.append(
            f"- C ratio score: {component_scores.get('c_ratio_score', 0):.2f} (40% weight)"
        )
        report.append(
            f"- Python interaction score: {component_scores.get('python_interaction_score', 0):.2f} (20% weight)"
        )
        report.append(
            f"- GIL operations score: {component_scores.get('gil_operations_score', 0):.2f} (10% weight)"
        )
        report.append(
            f"- Vectorizable loops score: {component_scores.get('vectorizable_loops_score', 0):.2f} (10% weight)"
        )
        report.append(
            f"- Feature usage score: {component_scores.get('feature_score', 0):.2f} (20% weight)"
        )
        report.append("")

    # Final score
    report.append(
        f"Overall optimization score: {analysis.get('optimization_score', 0):.2f}"
    )

    # Optimization suggestions
    line_categories = analysis.get("line_categories", {})
    if line_categories:
        report.append("\nOptimization Suggestions:")
        python_lines = [
            line for line, cat in line_categories.items() if cat == "python_interaction"
        ]
        gil_lines = [
            line for line, cat in line_categories.items() if cat == "gil_acquisition"
        ]
        loop_lines = [
            line for line, cat in line_categories.items() if cat == "vectorizable_loop"
        ]
        math_lines = [
            line for line, cat in line_categories.items() if cat == "unoptimized_math"
        ]

        if python_lines:
            report.append(
                f"- Lines with Python interaction (use C types): {python_lines[:5]}"
            )

        if gil_lines:
            report.append(f"- Lines with GIL acquisition (use nogil): {gil_lines[:5]}")

        if loop_lines:
            report.append(
                f"- Loops that could be vectorized (use prange): {loop_lines[:5]}"
            )

        if math_lines:
            report.append(
                f"- Unoptimized math operations (use C types): {math_lines[:5]}"
            )

    return "\n".join(report)


def get_analysis_explanation(analysis_result: CythonAnalysisResult) -> str:
    """
    Generate a human-readable explanation of the Cython analysis result.

    Args:
        analysis_result: The analysis result to explain

    Returns:
        str: A detailed explanation of the analysis
    """
    if not analysis_result.success:
        return f"Analysis failed: {analysis_result.error}"

    cs = analysis_result.component_scores

    explanation = ["Cython Optimization Analysis Report"]
    explanation.append("=" * 40)
    explanation.append("")

    explanation.append(
        f"Overall optimization score: {analysis_result.optimization_score:.2f}"
    )
    explanation.append(f"C code ratio: {analysis_result.c_ratio:.2%}")
    explanation.append(f"Python interaction ratio: {analysis_result.python_ratio:.2%}")
    explanation.append(f"Feature density: {analysis_result.feature_density:.3f}")
    explanation.append("")

    explanation.append("Optimization Score Components:")
    explanation.append(f"- C ratio score: {cs.c_ratio_score:.2f} (40% weight)")
    explanation.append(
        f"- Python interaction score: {cs.python_interaction_score:.2f} (20% weight)"
    )
    explanation.append(
        f"- GIL operations score: {cs.gil_operations_score:.2f} (10% weight)"
    )
    explanation.append(
        f"- Vectorizable loops score: {cs.vectorizable_loops_score:.2f} (10% weight)"
    )
    explanation.append(f"- Feature usage score: {cs.feature_score:.2f} (20% weight)")
    explanation.append("")

    explanation.append("Cython Features Used:")
    explanation.append(f"- C variable declarations (cdef): {analysis_result.cdef_vars}")
    explanation.append(f"- C functions (cdef): {analysis_result.cdef_funcs}")
    explanation.append(
        f"- Python-accessible C functions (cpdef): {analysis_result.cpdef_funcs}"
    )
    explanation.append(f"- Memory views: {analysis_result.memoryviews}")
    explanation.append(f"- Type-annotated arguments: {analysis_result.typed_args}")
    explanation.append(f"- No-GIL sections: {analysis_result.nogil}")
    explanation.append(f"- Parallel loops (prange): {analysis_result.prange}")
    explanation.append(f"- Cython directives: {analysis_result.directives}")

    if analysis_result.optimization_hints:
        explanation.append("\nOptimization Suggestions:")
        # Group hints by category to avoid repetition
        hint_categories = {}
        for hint in analysis_result.optimization_hints:
            if hint.category not in hint_categories:
                hint_categories[hint.category] = []
            hint_categories[hint.category].append(hint.line_number)

        for category, lines in hint_categories.items():
            hint_message = get_hint_message(category)
            line_str = ", ".join(str(line) for line in sorted(lines)[:5])
            if len(lines) > 5:
                line_str += f", ... ({len(lines)-5} more)"
            explanation.append(f"- {hint_message} (lines: {line_str})")

    return "\n".join(explanation)


def metrics_to_analysis_result(metrics: Dict[str, Any]) -> CythonAnalysisResult:
    """
    Convert dictionary-based metrics to a CythonAnalysisResult object.

    Args:
        metrics: Dictionary of metrics from CythonAnalyzer

    Returns:
        CythonAnalysisResult: Structured analysis result
    """
    # Create a structured analysis result
    analysis_result = CythonAnalysisResult(
        optimization_score=metrics.get("optimization_score", 0.0),
        c_ratio=metrics.get("c_ratio", 0.0),
        python_ratio=metrics.get("python_ratio", 0.0),
        feature_density=metrics.get("feature_density", 0.0),
        success=True if "error" not in metrics else False,
        error=metrics.get("error"),
    )

    # Set component scores
    component_scores = metrics.get("component_scores", {})
    analysis_result.component_scores = OptimizationScoreComponents(
        c_ratio_score=component_scores.get("c_ratio_score", 0.0),
        python_interaction_score=component_scores.get("python_interaction_score", 0.0),
        gil_operations_score=component_scores.get("gil_operations_score", 0.0),
        vectorizable_loops_score=component_scores.get("vectorizable_loops_score", 0.0),
        feature_score=component_scores.get("feature_score", 0.0),
    )

    # Set feature counts
    code_features = metrics.get("code_features", {})
    analysis_result.cdef_vars = code_features.get("cdef_vars", 0)
    analysis_result.cdef_funcs = code_features.get("cdef_funcs", 0)
    analysis_result.cpdef_funcs = code_features.get("cpdef_funcs", 0)
    analysis_result.memoryviews = code_features.get("memoryviews", 0)
    analysis_result.typed_args = code_features.get("typed_args", 0)
    analysis_result.nogil = code_features.get("nogil", 0)
    analysis_result.prange = code_features.get("prange", 0)
    analysis_result.directives = code_features.get("directives", 0)

    # Set optimization hints
    line_categories = metrics.get("line_categories", {})
    hints = []
    for line_num, category in line_categories.items():
        message = get_hint_message(category)
        hints.append(
            OptimizationHint(line_number=line_num, category=category, message=message)
        )
    analysis_result.optimization_hints = hints

    return analysis_result
