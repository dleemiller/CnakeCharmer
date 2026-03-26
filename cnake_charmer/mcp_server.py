"""
MCP server exposing CnakeCharmer validation tools.

These are the same tools used during GRPO training — compile, annotate,
test, benchmark, and composite_reward. Claude Code can call them to
iterate on Cython implementations while adding new problems.

Usage:
    claude mcp add cnake-charmer -- uv run python -m cnake_charmer.mcp_server
"""

import json
import logging

from mcp.server.fastmcp import FastMCP

from cnake_charmer.rewards.composite import composite_reward as _composite_reward
from cnake_charmer.validate.annotations import parse_annotations
from cnake_charmer.validate.benchmark import run_benchmark as _run_benchmark
from cnake_charmer.validate.compiler import cleanup_build, compile_cython
from cnake_charmer.validate.correctness import _load_module_from_path, check_correctness

logger = logging.getLogger(__name__)

mcp = FastMCP("cnake-charmer")


@mcp.tool()
def compile_check(code: str) -> str:
    """Compile Cython (.pyx) code and check for errors.

    Args:
        code: Complete .pyx source code to compile.

    Returns:
        JSON with success status and any error messages.
    """
    result = compile_cython(code, annotate=False)
    output = {"success": result.success, "errors": result.errors, "warnings": result.warnings}
    cleanup_build(result)
    return json.dumps(output, indent=2)


@mcp.tool()
def annotate_cython(code: str) -> str:
    """Compile Cython code and analyze HTML annotations for optimization quality.

    Returns a score (0.0 = all Python, 1.0 = all C) and hints about
    lines that fall back to Python object interactions.

    Args:
        code: Complete .pyx source code to analyze.

    Returns:
        JSON with score, yellow/white line counts, and optimization hints.
    """
    result = compile_cython(code, annotate=True, keep_build=True)

    if not result.success:
        output = {"success": False, "errors": result.errors, "score": 0.0, "hints": []}
        cleanup_build(result)
        return json.dumps(output, indent=2)

    ann = parse_annotations(html_path=result.html_path) if result.html_path else None
    cleanup_build(result)

    if ann and ann.success:
        return json.dumps(
            {
                "success": True,
                "score": round(ann.score, 3),
                "total_lines": ann.total_lines,
                "white_lines": ann.white_lines,
                "yellow_lines": ann.yellow_lines,
                "hints": ann.hints,
            },
            indent=2,
        )

    return json.dumps({"success": True, "score": 0.0, "hints": ["Could not parse annotations"]})


@mcp.tool()
def test_correctness(cython_code: str, python_code: str, func_name: str, test_inputs: str) -> str:
    """Test a Cython implementation against a Python reference for correctness.

    Compiles the Cython code, runs both implementations with the same inputs,
    and compares outputs.

    Args:
        cython_code: Complete .pyx source code.
        python_code: Python reference implementation (will be exec'd).
        func_name: Name of the function to test in both implementations.
        test_inputs: JSON array of test inputs, e.g. '[[10], [20], [50]]'.
            Each entry is a list of positional args.

    Returns:
        JSON with passed/total counts and any failure details.
    """
    # Parse test inputs
    try:
        inputs = json.loads(test_inputs)
        test_cases = [((tuple(tc),) if isinstance(tc, list) else ((tc,),)) for tc in inputs]
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"success": False, "error": f"Invalid test_inputs JSON: {e}"})

    # Load Python function
    namespace = {}
    try:
        exec(python_code, namespace)  # noqa: S102
        py_func = namespace[func_name]
    except Exception as e:
        return json.dumps({"success": False, "error": f"Failed to load Python function: {e}"})

    # Compile Cython
    comp = compile_cython(cython_code, annotate=False, keep_build=True)
    if not comp.success:
        output = {"success": False, "error": comp.errors, "passed": 0, "total": len(test_cases)}
        cleanup_build(comp)
        return json.dumps(output, indent=2)

    try:
        module = _load_module_from_path(comp.module_path, "gen_module")
        cy_func = getattr(module, func_name)
    except Exception as e:
        cleanup_build(comp)
        return json.dumps({"success": False, "error": f"Failed to load Cython function: {e}"})

    result = check_correctness(python_func=py_func, cython_func=cy_func, test_cases=test_cases)
    cleanup_build(comp)

    return json.dumps(
        {
            "success": True,
            "passed": result.passed,
            "total": result.total,
            "score": result.score,
            "failures": result.failures,
        },
        indent=2,
    )


@mcp.tool()
def benchmark_cython(
    cython_code: str, python_code: str, func_name: str, benchmark_args: str
) -> str:
    """Benchmark a Cython implementation against a Python reference.

    Measures execution time of both and computes the speedup ratio.

    Args:
        cython_code: Complete .pyx source code.
        python_code: Python reference implementation (will be exec'd).
        func_name: Name of the function to benchmark.
        benchmark_args: JSON array of positional args, e.g. '[10000]'.

    Returns:
        JSON with speedup ratio and timing details.
    """
    try:
        args = tuple(json.loads(benchmark_args))
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"success": False, "error": f"Invalid benchmark_args JSON: {e}"})

    namespace = {}
    try:
        exec(python_code, namespace)  # noqa: S102
        py_func = namespace[func_name]
    except Exception as e:
        return json.dumps({"success": False, "error": f"Failed to load Python function: {e}"})

    comp = compile_cython(cython_code, annotate=False, keep_build=True)
    if not comp.success:
        cleanup_build(comp)
        return json.dumps({"success": False, "error": comp.errors})

    try:
        module = _load_module_from_path(comp.module_path, "gen_module")
        cy_func = getattr(module, func_name)
    except Exception as e:
        cleanup_build(comp)
        return json.dumps({"success": False, "error": f"Failed to load function: {e}"})

    result = _run_benchmark(python_func=py_func, cython_func=cy_func, args=args, num_runs=5)
    cleanup_build(comp)

    return json.dumps(
        {
            "success": result.success,
            "speedup": round(result.speedup, 2),
            "python_ms": round(result.python_time * 1000, 3),
            "cython_ms": round(result.cython_time * 1000, 3),
        },
        indent=2,
    )


@mcp.tool()
def score_cython(
    cython_code: str, python_code: str, func_name: str, test_inputs: str, benchmark_args: str
) -> str:
    """Run the full composite reward on a Cython implementation.

    This is the same reward function used during GRPO training.
    Evaluates compilation, correctness, speedup, and annotation quality.

    Args:
        cython_code: Complete .pyx source code.
        python_code: Python reference implementation (will be exec'd).
        func_name: Name of the function.
        test_inputs: JSON array of test inputs, e.g. '[[10], [20], [50]]'.
        benchmark_args: JSON array of benchmark args, e.g. '[10000]'.

    Returns:
        JSON with compiled, correctness, speedup, annotation score, and total reward.
    """
    try:
        inputs = json.loads(test_inputs)
        test_cases = [((tuple(tc),) if isinstance(tc, list) else ((tc,),)) for tc in inputs]
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid test_inputs: {e}"})

    try:
        b_args = tuple(json.loads(benchmark_args))
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid benchmark_args: {e}"})

    namespace = {}
    try:
        exec(python_code, namespace)  # noqa: S102
        py_func = namespace[func_name]
    except Exception as e:
        return json.dumps({"error": f"Failed to load Python function: {e}"})

    scores = _composite_reward(
        cython_code=cython_code,
        python_func=py_func,
        func_name=func_name,
        test_cases=test_cases,
        benchmark_args=b_args,
        benchmark_runs=3,
    )

    # Clean up for JSON serialization
    output = {
        "compiled": scores["compiled"],
        "correctness": scores["correctness"],
        "speedup": round(scores["speedup"], 2),
        "annotation_score": round(scores["annotations"], 3),
        "total_reward": round(scores["total"], 3),
        "annotation_hints": scores["annotation_hints"],
        "correctness_failures": scores["correctness_failures"],
        "compilation_errors": scores["compilation_errors"],
    }
    return json.dumps(output, indent=2)


if __name__ == "__main__":
    mcp.run()
