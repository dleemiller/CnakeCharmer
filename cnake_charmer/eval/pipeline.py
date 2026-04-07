"""
Validation pipeline that orchestrates compilation, correctness,
benchmarking, and annotation analysis.

All code execution (correctness, benchmark) runs in a bwrap sandbox.
The pipeline never loads untrusted .so files into its own process.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from cnake_charmer.eval.annotations import AnnotationResult, parse_annotations
from cnake_charmer.eval.benchmark import BenchmarkResult, run_benchmark
from cnake_charmer.eval.compiler import CompilationResult, cleanup_build, compile_cython
from cnake_charmer.eval.correctness import (
    CorrectnessResult,
    _load_module_from_path,
    check_correctness,
)
from cnake_charmer.eval.lint import LintResult, run_cython_lint
from cnake_charmer.eval.memory_safety import MemorySafetyResult, check_memory_safety

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    compilation: CompilationResult | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkResult | None = None
    annotations: AnnotationResult | None = None
    lint: LintResult | None = None
    memory_safety: MemorySafetyResult | None = None

    @property
    def compiled(self) -> bool:
        return self.compilation is not None and self.compilation.success

    @property
    def correct(self) -> bool:
        return self.correctness is not None and self.correctness.score == 1.0

    @property
    def speedup(self) -> float:
        if self.benchmark and self.benchmark.success:
            return self.benchmark.speedup
        return 0.0

    @property
    def annotation_score(self) -> float:
        if self.annotations and self.annotations.success:
            return self.annotations.score
        return 0.0

    @property
    def lint_score(self) -> float:
        if self.lint and self.lint.success:
            return self.lint.score
        return 1.0  # don't penalize if lint unavailable

    @property
    def memory_safety_score(self) -> float:
        if self.memory_safety and self.memory_safety.success:
            return self.memory_safety.score
        return 1.0  # don't penalize if ASan unavailable

    def summary(self) -> dict:
        """Return a flat summary dict suitable for training data."""
        return {
            "compiled": self.compiled,
            "compilation_errors": self.compilation.errors if self.compilation else "",
            "correctness_score": self.correctness.score if self.correctness else 0.0,
            "correctness_failures": self.correctness.failures if self.correctness else [],
            "speedup": self.speedup,
            "annotation_score": self.annotation_score,
            "annotation_hints": self.annotations.hints if self.annotations else [],
            "lint_score": self.lint_score,
            "lint_violations": self.lint.violations if self.lint else [],
            "memory_safety_score": self.memory_safety_score,
            "memory_safety_errors": self.memory_safety.errors if self.memory_safety else [],
        }


def validate(
    cython_code: str,
    python_func: Callable | None = None,
    func_name: str | None = None,
    test_cases: list | None = None,
    benchmark_args: tuple | None = None,
    benchmark_kwargs: dict | None = None,
    benchmark_runs: int = 10,
    skip_benchmark: bool = False,
    skip_correctness: bool = False,
    skip_memory_safety: bool = False,
    module_name: str = "gen_module",
    python_code: str | None = None,
) -> ValidationResult:
    """
    Run the full validation pipeline on a Cython code string.

    Pipeline: compile -> lint -> annotate -> correctness -> benchmark -> ASan

    When python_code (source string) is provided, correctness and benchmark
    run in a sandboxed subprocess. When only python_func (callable) is
    provided, falls back to legacy in-process execution.

    Args:
        cython_code: The .pyx source code as a string.
        python_func: Reference Python callable (legacy, in-process).
        python_code: Python source code string (sandboxed, preferred).
        func_name: Name of the function to extract from the compiled module.
        test_cases: Test cases for correctness checking.
        benchmark_args: Args for benchmarking (defaults to first test case args).
        benchmark_kwargs: Kwargs for benchmarking.
        benchmark_runs: Number of benchmark iterations.
        skip_benchmark: Skip the benchmark step.
        skip_correctness: Skip the correctness step.
        skip_memory_safety: Skip the memory safety step.
        module_name: Name for the compiled module.

    Returns:
        ValidationResult with all sub-results.
    """
    result = ValidationResult()

    # Step 1: Compile (always, with annotations)
    result.compilation = compile_cython(
        cython_code,
        module_name=module_name,
        annotate=True,
        keep_build=True,  # Need the build dir for subsequent steps
    )

    if not result.compilation.success:
        logger.debug(f"Compilation failed, stopping pipeline: {result.compilation.errors[:200]}")
        cleanup_build(result.compilation)
        return result

    try:
        # Step 2: Lint analysis (runs on source, independent of compilation)
        result.lint = run_cython_lint(cython_code)

        # Step 3: Parse annotations (from compilation HTML)
        if result.compilation.html_path:
            result.annotations = parse_annotations(html_path=result.compilation.html_path)

        module_path = result.compilation.module_path

        # Determine execution mode: sandboxed (preferred) or legacy in-process
        use_sandbox = python_code and func_name and module_path

        # Legacy path: load the compiled function in-process (backward compat)
        cython_func = None
        if not use_sandbox and func_name and module_path:
            try:
                module = _load_module_from_path(module_path, module_name)
                cython_func = getattr(module, func_name)
            except Exception as e:
                logger.warning(f"Could not load function '{func_name}' from compiled module: {e}")

        # Step 4: Correctness check
        if not skip_correctness and test_cases:
            if use_sandbox:
                result.correctness = check_correctness(
                    python_code=python_code,
                    func_name=func_name,
                    cython_module_path=module_path,
                    test_cases=test_cases,
                )
            elif python_func and cython_func:
                result.correctness = check_correctness(
                    python_func=python_func,
                    cython_func=cython_func,
                    test_cases=test_cases,
                )

        # Step 5: Benchmark
        b_args = benchmark_args
        if b_args is None and test_cases:
            b_args = _extract_benchmark_args(test_cases)

        if not skip_benchmark and b_args is not None:
            if use_sandbox:
                result.benchmark = run_benchmark(
                    python_code=python_code,
                    func_name=func_name,
                    cython_module_path=module_path,
                    args=b_args,
                    kwargs=benchmark_kwargs,
                    num_runs=benchmark_runs,
                )
            elif python_func and cython_func:
                result.benchmark = run_benchmark(
                    python_func=python_func,
                    cython_func=cython_func,
                    args=b_args,
                    kwargs=benchmark_kwargs,
                    num_runs=benchmark_runs,
                )

        # Step 6: Memory safety (ASan) — uses small test args
        if not skip_memory_safety and func_name:
            asan_args = benchmark_args
            if asan_args is None and test_cases:
                asan_args = _extract_benchmark_args(test_cases)

            if asan_args is not None:
                small_args = _shrink_args(asan_args)
                result.memory_safety = check_memory_safety(
                    cython_code=cython_code,
                    func_name=func_name,
                    test_args=small_args,
                )
    finally:
        cleanup_build(result.compilation)

    return result


def _extract_benchmark_args(test_cases: list) -> tuple | None:
    """Extract benchmark args from the first test case."""
    if not test_cases:
        return None
    case = test_cases[0]
    if isinstance(case, dict):
        return tuple(case.get("args", ()))
    elif isinstance(case, (list, tuple)) and len(case) > 0:
        return tuple(case[0]) if isinstance(case[0], (list, tuple)) else tuple(case)
    else:
        return (case,) if case is not None else ()


def _shrink_args(args: tuple) -> tuple:
    """Shrink benchmark args to small values for fast ASan checking."""
    shrunk = []
    for a in args:
        if isinstance(a, int) and a > 100:
            shrunk.append(min(a, 100))
        else:
            shrunk.append(a)
    return tuple(shrunk)


# ---------------------------------------------------------------------------
# Composite scoring (absorbed from rewards/composite.py)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "correctness": 0.30,
    "performance": 0.25,
    "annotations": 0.20,
    "lint": 0.10,
    "memory_safety": 0.15,
}


def composite_reward(
    cython_code: str,
    python_func: Callable | None = None,
    func_name: str = "",
    test_cases: list | None = None,
    benchmark_args: tuple | None = None,
    benchmark_runs: int = 5,
    weights: dict | None = None,
    python_code: str | None = None,
    **kwargs,
) -> dict:
    """
    Compute the full composite reward.

    Returns a dict with individual scores and the weighted total.
    Compilation is a gate — 0 total if it fails.

    When python_code (source string) is provided, execution runs in
    a sandboxed subprocess for safety.
    """
    w = weights or DEFAULT_WEIGHTS

    result = validate(
        cython_code=cython_code,
        python_func=python_func,
        python_code=python_code,
        func_name=func_name,
        test_cases=test_cases,
        benchmark_args=benchmark_args,
        benchmark_runs=benchmark_runs,
    )

    scores = {
        "compiled": result.compiled,
        "compilation_errors": result.compilation.errors if result.compilation else "",
        "correctness": 0.0,
        "performance": 0.0,
        "annotations": 0.0,
        "lint": 0.0,
        "memory_safety": 0.0,
        "speedup": 0.0,
        "correctness_failures": [],
        "annotation_hints": [],
        "lint_violations": [],
        "memory_safety_errors": [],
        "total": 0.0,
    }

    if not result.compiled:
        return scores

    if result.correctness is not None:
        scores["correctness"] = result.correctness.score
        scores["correctness_failures"] = result.correctness.failures

    if result.benchmark is not None and result.benchmark.success:
        speedup = result.benchmark.speedup
        scores["speedup"] = speedup
        if speedup > 1.0:
            scores["performance"] = min(math.log2(speedup) / math.log2(10), 1.0)

    if result.annotations is not None and result.annotations.success:
        scores["annotations"] = result.annotations.score
        scores["annotation_hints"] = result.annotations.hints

    if result.lint is not None and result.lint.success:
        scores["lint"] = result.lint.score
        scores["lint_violations"] = result.lint.violations

    if result.memory_safety is not None and result.memory_safety.success:
        scores["memory_safety"] = result.memory_safety.score
        scores["memory_safety_errors"] = result.memory_safety.errors
    else:
        scores["memory_safety"] = 1.0

    scores["total"] = (
        w.get("correctness", 0.30) * scores["correctness"]
        + w.get("performance", 0.25) * scores["performance"]
        + w.get("annotations", 0.20) * scores["annotations"]
        + w.get("lint", 0.10) * scores["lint"]
        + w.get("memory_safety", 0.15) * scores["memory_safety"]
    )

    return scores
