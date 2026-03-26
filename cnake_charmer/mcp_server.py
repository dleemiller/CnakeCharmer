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
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from cnake_charmer.dataset.loader import discover_pairs
from cnake_charmer.rewards.composite import composite_reward as _composite_reward
from cnake_charmer.validate.annotations import parse_annotations
from cnake_charmer.validate.compiler import cleanup_build, compile_cython

logger = logging.getLogger(__name__)

mcp = FastMCP("cnake-charmer")

PACKAGE_ROOT = Path(__file__).parent
PY_DIR = PACKAGE_ROOT / "py"
CY_DIR = PACKAGE_ROOT / "cy"


# ---------------------------------------------------------------------------
# Primary tool: score a problem by name (reads files from repo)
# ---------------------------------------------------------------------------


@mcp.tool()
def score_problem(problem_id: str) -> str:
    """Score a Python/Cython problem pair by its ID (e.g. 'numerical/great_circle').

    Reads the .py and .pyx files from the repo, extracts test cases from
    the test file, and runs the full composite reward: compilation,
    correctness, speedup, and annotation quality.

    Args:
        problem_id: Problem path like 'numerical/great_circle' or 'algorithms/primes'.

    Returns:
        JSON with compiled, correctness, speedup, annotation score, total reward,
        and actionable hints for improvement.
    """
    pairs = discover_pairs()
    match = None
    for p in pairs:
        if p.problem_id == problem_id:
            match = p
            break

    if match is None:
        available = sorted(p.problem_id for p in pairs)
        return json.dumps(
            {
                "error": f"Problem '{problem_id}' not found",
                "available": available,
            },
            indent=2,
        )

    if not match.has_cython:
        return json.dumps({"error": f"No Cython implementation found for '{problem_id}'"})

    # Exec the Python function
    namespace = {}
    try:
        exec(match.python_code, namespace)  # noqa: S102
        py_func = namespace.get(match.func_name)
    except Exception as e:
        return json.dumps({"error": f"Failed to load Python function: {e}"})

    if py_func is None:
        return json.dumps({"error": f"Function '{match.func_name}' not found in Python code"})

    scores = _composite_reward(
        cython_code=match.cython_code,
        python_func=py_func,
        func_name=match.func_name,
        test_cases=match.test_cases,
        benchmark_args=match.benchmark_args,
        benchmark_runs=3,
    )

    return json.dumps(
        {
            "problem_id": problem_id,
            "func_name": match.func_name,
            "compiled": scores["compiled"],
            "correctness": scores["correctness"],
            "speedup": round(scores["speedup"], 2),
            "annotation_score": round(scores["annotations"], 3),
            "total_reward": round(scores["total"], 3),
            "annotation_hints": scores["annotation_hints"],
            "correctness_failures": scores["correctness_failures"],
            "compilation_errors": scores["compilation_errors"],
        },
        indent=2,
    )


@mcp.tool()
def list_problems() -> str:
    """List all problem pairs in the dataset with their status.

    Returns:
        JSON array of problems with id, func_name, has_cython, category, and test count.
    """
    pairs = discover_pairs()
    result = []
    for p in sorted(pairs, key=lambda x: x.problem_id):
        result.append(
            {
                "problem_id": p.problem_id,
                "func_name": p.func_name,
                "has_cython": p.has_cython,
                "category": p.category,
                "num_tests": len(p.test_cases),
                "benchmark_args": list(p.benchmark_args) if p.benchmark_args else None,
            }
        )
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Code-level tools (for iterating on implementations)
# ---------------------------------------------------------------------------


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
def score_cython(
    cython_code: str, python_code: str, func_name: str, test_inputs: str, benchmark_args: str
) -> str:
    """Run the full composite reward on Cython code against a Python reference.

    Use score_problem() instead when the files already exist in the repo.
    This tool is for scoring code that hasn't been saved to files yet.

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

    return json.dumps(
        {
            "compiled": scores["compiled"],
            "correctness": scores["correctness"],
            "speedup": round(scores["speedup"], 2),
            "annotation_score": round(scores["annotations"], 3),
            "total_reward": round(scores["total"], 3),
            "annotation_hints": scores["annotation_hints"],
            "correctness_failures": scores["correctness_failures"],
            "compilation_errors": scores["compilation_errors"],
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
