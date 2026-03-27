"""
Validation pipeline that orchestrates compilation, correctness,
benchmarking, and annotation analysis.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from cnake_charmer.rewards.lint import LintResult, run_cython_lint
from cnake_charmer.validate.annotations import AnnotationResult, parse_annotations
from cnake_charmer.validate.benchmark import BenchmarkResult, run_benchmark
from cnake_charmer.validate.compiler import CompilationResult, cleanup_build, compile_cython
from cnake_charmer.validate.correctness import (
    CorrectnessResult,
    _load_module_from_path,
    check_correctness,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    compilation: CompilationResult | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkResult | None = None
    annotations: AnnotationResult | None = None
    lint: LintResult | None = None

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
    module_name: str = "gen_module",
) -> ValidationResult:
    """
    Run the full validation pipeline on a Cython code string.

    Pipeline: compile -> correctness -> benchmark -> annotate

    Args:
        cython_code: The .pyx source code as a string.
        python_func: Reference Python function for correctness/benchmark.
        func_name: Name of the function to extract from the compiled module.
        test_cases: Test cases for correctness checking.
        benchmark_args: Args for benchmarking (defaults to first test case args).
        benchmark_kwargs: Kwargs for benchmarking.
        benchmark_runs: Number of benchmark iterations.
        skip_benchmark: Skip the benchmark step.
        skip_correctness: Skip the correctness step.
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
        return result

    # Step 2: Lint analysis (runs on source, independent of compilation)
    result.lint = run_cython_lint(cython_code)

    # Step 3: Parse annotations (from compilation HTML)
    if result.compilation.html_path:
        result.annotations = parse_annotations(html_path=result.compilation.html_path)

    # Load the compiled function for correctness and benchmarking
    cython_func = None
    if func_name and result.compilation.module_path:
        try:
            module = _load_module_from_path(result.compilation.module_path, module_name)
            cython_func = getattr(module, func_name)
        except Exception as e:
            logger.warning(f"Could not load function '{func_name}' from compiled module: {e}")

    # Step 4: Correctness check
    if not skip_correctness and python_func and cython_func and test_cases:
        result.correctness = check_correctness(
            python_func=python_func,
            cython_func=cython_func,
            test_cases=test_cases,
        )

    # Step 5: Benchmark
    if not skip_benchmark and python_func and cython_func:
        b_args = benchmark_args
        if b_args is None and test_cases:
            # Use first test case as benchmark args
            case = test_cases[0]
            if isinstance(case, dict):
                b_args = tuple(case.get("args", ()))
            elif isinstance(case, (list, tuple)) and len(case) > 0:
                b_args = tuple(case[0]) if isinstance(case[0], (list, tuple)) else tuple(case)
            else:
                b_args = (case,) if case is not None else ()

        if b_args is not None:
            result.benchmark = run_benchmark(
                python_func=python_func,
                cython_func=cython_func,
                args=b_args,
                kwargs=benchmark_kwargs,
                num_runs=benchmark_runs,
            )

    # Clean up build directory
    cleanup_build(result.compilation)

    return result
