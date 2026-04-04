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
from cnake_charmer.eval.annotations import parse_annotations
from cnake_charmer.eval.compiler import cleanup_build, compile_cython
from cnake_charmer.eval.memory_safety import check_memory_safety
from cnake_charmer.eval.pipeline import composite_reward as _composite_reward

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
            "lint_score": round(scores.get("lint", 0.0), 3),
            "memory_safety_score": round(scores.get("memory_safety", 1.0), 3),
            "total_reward": round(scores["total"], 3),
            "annotation_hints": scores["annotation_hints"],
            "lint_violations": scores.get("lint_violations", []),
            "memory_safety_errors": scores.get("memory_safety_errors", []),
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
# evaluate_cython: same interface as the training tool
# ---------------------------------------------------------------------------


@mcp.tool()
def evaluate_cython(code: str, python_code: str, test_code: str) -> str:
    """Compile Cython code, test equivalence against Python reference, and benchmark.

    The test_code runs in a namespace where `py` is the Python module and
    `cy` is the compiled Cython module. Each test assertion has a 5-second timeout.

    Args:
        code: Complete .pyx Cython source code.
        python_code: Original Python source code (reference implementation).
        test_code: Equivalence test assertions comparing py.<name>(...) == cy.<name>(...).

    Returns:
        Compilation status, annotation score, test results, and benchmark speedup.
    """
    from cnake_charmer.training.environment import CythonToolEnvironment

    env = CythonToolEnvironment()
    env.reset()
    return env.evaluate_cython(code=code, python_code=python_code, test_code=test_code)


# ---------------------------------------------------------------------------
# File-based tools (for iterating on implementations)
# ---------------------------------------------------------------------------

SIMD_FLAGS = ["-mavx2", "-mfma", "-O3"]


def _detect_compile_flags(pyx_path: str) -> list:
    """Auto-detect compiler flags from file path.

    Files in cy_simd/ or nn_ops/ get SIMD flags automatically.
    """
    if "cy_simd" in pyx_path or "nn_ops" in pyx_path:
        return SIMD_FLAGS
    return []


@mcp.tool()
def compile_file(pyx_path: str) -> str:
    """Compile a .pyx file and check for errors.

    Auto-detects SIMD flags for cy_simd/ and nn_ops/ files.

    Args:
        pyx_path: Path to a .pyx file, e.g. 'cnake_charmer/cy_simd/nn_ops/relu.pyx'.

    Returns:
        JSON with success status and any error messages.
    """
    path = Path(pyx_path)
    if not path.exists():
        return json.dumps({"success": False, "errors": f"File not found: {pyx_path}"})

    flags = _detect_compile_flags(pyx_path)
    code = path.read_text()
    result = compile_cython(code, annotate=False, extra_compile_args=flags)
    output = {
        "success": result.success,
        "errors": result.errors,
        "warnings": result.warnings,
        "flags": flags,
    }
    cleanup_build(result)
    return json.dumps(output, indent=2)


@mcp.tool()
def annotate_file(pyx_path: str) -> str:
    """Compile a .pyx file and analyze HTML annotations for optimization quality.

    Auto-detects SIMD flags for cy_simd/ and nn_ops/ files.

    Args:
        pyx_path: Path to a .pyx file, e.g. 'cnake_charmer/cy/numerical/great_circle.pyx'.

    Returns:
        JSON with score, yellow/white line counts, and optimization hints.
    """
    path = Path(pyx_path)
    if not path.exists():
        return json.dumps({"success": False, "errors": f"File not found: {pyx_path}"})

    flags = _detect_compile_flags(pyx_path)
    code = path.read_text()
    result = compile_cython(code, annotate=True, keep_build=True, extra_compile_args=flags)

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
def check_memory(pyx_path: str, func_name: str, test_args: str = "(100,)") -> str:
    """Run AddressSanitizer on a .pyx file to detect memory errors.

    Compiles with -fsanitize=address and runs the function with small inputs.
    Detects leaks, buffer overflows, use-after-free, and double-free.

    Args:
        pyx_path: Path to a .pyx file.
        func_name: Name of the function to test.
        test_args: Python tuple literal for test arguments, e.g. '(100,)'.

    Returns:
        JSON with score (1.0 = clean, 0.0 = errors), error details, and leak bytes.
    """
    path = Path(pyx_path)
    if not path.exists():
        return json.dumps({"success": False, "errors": [f"File not found: {pyx_path}"]})

    try:
        args = eval(test_args)  # noqa: S307
        if not isinstance(args, tuple):
            args = (args,)
    except Exception as e:
        return json.dumps({"success": False, "errors": [f"Invalid test_args: {e}"]})

    code = path.read_text()
    flags = _detect_compile_flags(pyx_path)
    result = check_memory_safety(
        cython_code=code,
        func_name=func_name,
        test_args=args,
        extra_compile_args=flags,
    )

    return json.dumps(
        {
            "success": result.success,
            "score": result.score,
            "error_count": result.error_count,
            "leak_bytes": result.leak_bytes,
            "error_types": result.error_types,
            "errors": result.errors,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
