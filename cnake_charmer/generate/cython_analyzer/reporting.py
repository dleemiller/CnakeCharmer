"""
Reporting utilities for Cython code analysis.
"""


def get_optimization_hints(metrics):
    """
    Generate optimization hints based on the analysis metrics.

    Args:
        metrics: Analysis metrics

    Returns:
        dict: Line number -> optimization hint mappings
    """
    hints = {}
    line_categories = metrics.get("line_categories", {})

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


def get_optimization_report(metrics):
    """
    Generate a detailed report of the optimization metrics.

    Args:
        metrics: Analysis metrics

    Returns:
        str: Detailed optimization report
    """
    if "error" in metrics:
        return f"Analysis failed: {metrics['error']}"

    report = ["Cython Optimization Analysis Report"]
    report.append("=" * 40)
    report.append("")

    # Basic metrics
    report.append(f"Total lines: {metrics.get('total_lines', 0)}")
    report.append(
        f"C code lines: {metrics.get('c_lines', 0)} ({metrics.get('c_ratio', 0):.2%})"
    )
    report.append(
        f"Python interaction lines: {metrics.get('python_lines', 0)} ({metrics.get('python_ratio', 0):.2%})"
    )
    report.append(f"GIL operations: {metrics.get('gil_operations', 0)}")
    report.append(f"Vectorizable loops: {metrics.get('vectorizable_loops', 0)}")
    report.append(f"Unoptimized math operations: {metrics.get('unoptimized_math', 0)}")
    report.append("")

    # Cython features
    code_features = metrics.get("code_features", {})
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
    component_scores = metrics.get("component_scores", {})
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
        f"Overall optimization score: {metrics.get('optimization_score', 0):.2f}"
    )

    # Optimization suggestions
    line_categories = metrics.get("line_categories", {})
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
