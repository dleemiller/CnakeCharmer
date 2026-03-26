"""
OpenEnv-compatible tool environment for TRL GRPOTrainer.

Exposes Cython validation tools (compile, annotate, test, benchmark)
as callable methods that TRL auto-discovers via docstring parsing.
The trainer handles the multi-turn loop: generate → tool call → execute → repeat.
"""

import json
import logging

from cnake_charmer.rewards.composite import composite_reward
from cnake_charmer.training.prompts import format_feedback
from cnake_charmer.validate.annotations import parse_annotations
from cnake_charmer.validate.benchmark import run_benchmark as _run_benchmark
from cnake_charmer.validate.compiler import cleanup_build, compile_cython
from cnake_charmer.validate.correctness import _load_module_from_path, check_correctness

logger = logging.getLogger(__name__)


def _exec_func(python_code: str, func_name: str):
    """Execute Python code string and extract the named function."""
    if not func_name or not python_code:
        return None
    namespace = {}
    try:
        exec(python_code, namespace)  # noqa: S102
        return namespace.get(func_name)
    except Exception as e:
        logger.warning(f"Failed to exec Python for {func_name}: {e}")
        return None


class CythonToolEnvironment:
    """Environment for training a Cython code generation agent.

    Provides tools to compile, test, benchmark, and analyze Cython code.
    Used with TRL GRPOTrainer's environment_factory parameter.
    """

    def reset(
        self,
        *,
        python_code: str = "",
        func_name: str = "",
        test_cases: str = "[]",
        benchmark_args: str = "null",
        **kwargs,
    ) -> str | None:
        """Reset the environment for a new episode.

        Called by TRL at the start of each rollout. Receives fields from the
        dataset row as keyword arguments.
        """
        self._python_code = python_code
        self._func_name = func_name
        self._python_func = _exec_func(python_code, func_name)
        self._test_cases = json.loads(test_cases) if isinstance(test_cases, str) else test_cases
        self._benchmark_args = (
            json.loads(benchmark_args) if isinstance(benchmark_args, str) else benchmark_args
        )
        if self._benchmark_args is not None and not isinstance(self._benchmark_args, tuple):
            self._benchmark_args = tuple(self._benchmark_args) if self._benchmark_args else None

        self.last_code = None
        self.last_scores = None
        return None

    def compile(self, code: str) -> str:
        """Compile Cython code and check for errors.

        Args:
            code: The complete .pyx Cython source code to compile.

        Returns:
            Compilation result with success status and any error messages.
        """
        self.last_code = code
        result = compile_cython(code, annotate=False)
        output = format_feedback("compile", {"success": result.success, "errors": result.errors})
        cleanup_build(result)
        return output

    def annotate(self, code: str) -> str:
        """Compile Cython code and analyze optimization quality via HTML annotations.

        Args:
            code: The complete .pyx Cython source code to analyze.

        Returns:
            Annotation score and optimization hints about Python-fallback lines.
        """
        self.last_code = code
        result = compile_cython(code, annotate=True, keep_build=True)

        if not result.success:
            output = format_feedback(
                "annotate",
                {
                    "success": False,
                    "errors": result.errors,
                    "score": 0.0,
                    "hints": [],
                    "yellow_lines": 0,
                    "total_lines": 0,
                },
            )
            cleanup_build(result)
            return output

        ann = parse_annotations(html_path=result.html_path) if result.html_path else None
        cleanup_build(result)

        if ann and ann.success:
            return format_feedback(
                "annotate",
                {
                    "success": True,
                    "score": ann.score,
                    "hints": ann.hints,
                    "yellow_lines": ann.yellow_lines,
                    "total_lines": ann.total_lines,
                },
            )

        return format_feedback(
            "annotate",
            {
                "success": True,
                "score": 0.0,
                "hints": ["Could not parse annotations"],
                "yellow_lines": 0,
                "total_lines": 0,
            },
        )

    def test(self, code: str) -> str:
        """Compile and test Cython code for correctness against the Python reference.

        Args:
            code: The complete .pyx Cython source code to test.

        Returns:
            Test results with pass/fail counts and failure details.
        """
        self.last_code = code

        if not self._python_func or not self._func_name or not self._test_cases:
            return format_feedback(
                "test",
                {
                    "success": False,
                    "errors": "No reference function or test cases",
                    "passed": 0,
                    "total": 0,
                    "failures": [],
                },
            )

        comp = compile_cython(code, annotate=False, keep_build=True)
        if not comp.success:
            output = format_feedback(
                "test",
                {
                    "success": False,
                    "errors": comp.errors,
                    "passed": 0,
                    "total": len(self._test_cases),
                    "failures": [],
                },
            )
            cleanup_build(comp)
            return output

        try:
            module = _load_module_from_path(comp.module_path, "gen_module")
            cython_func = getattr(module, self._func_name)
        except Exception as e:
            cleanup_build(comp)
            return format_feedback(
                "test",
                {
                    "success": False,
                    "errors": f"Could not load function: {e}",
                    "passed": 0,
                    "total": len(self._test_cases),
                    "failures": [],
                },
            )

        result = check_correctness(
            python_func=self._python_func,
            cython_func=cython_func,
            test_cases=self._test_cases,
        )
        cleanup_build(comp)

        return format_feedback(
            "test",
            {
                "success": True,
                "passed": result.passed,
                "total": result.total,
                "failures": result.failures,
            },
        )

    def benchmark(self, code: str) -> str:
        """Compile and benchmark Cython code against the Python reference.

        Args:
            code: The complete .pyx Cython source code to benchmark.

        Returns:
            Speedup ratio and timing details compared to the Python implementation.
        """
        self.last_code = code

        if not self._python_func or not self._func_name:
            return format_feedback(
                "benchmark",
                {
                    "success": False,
                    "errors": "No reference function",
                    "speedup": 0.0,
                    "cython_time": 0.0,
                    "python_time": 0.0,
                },
            )

        comp = compile_cython(code, annotate=False, keep_build=True)
        if not comp.success:
            output = format_feedback(
                "benchmark",
                {
                    "success": False,
                    "errors": comp.errors,
                    "speedup": 0.0,
                    "cython_time": 0.0,
                    "python_time": 0.0,
                },
            )
            cleanup_build(comp)
            return output

        try:
            module = _load_module_from_path(comp.module_path, "gen_module")
            cython_func = getattr(module, self._func_name)
        except Exception as e:
            cleanup_build(comp)
            return format_feedback(
                "benchmark",
                {
                    "success": False,
                    "errors": f"Could not load function: {e}",
                    "speedup": 0.0,
                    "cython_time": 0.0,
                    "python_time": 0.0,
                },
            )

        b_args = self._benchmark_args or ()
        result = _run_benchmark(
            python_func=self._python_func,
            cython_func=cython_func,
            args=b_args,
            num_runs=3,
        )
        cleanup_build(comp)

        return format_feedback(
            "benchmark",
            {
                "success": result.success,
                "speedup": round(result.speedup, 2),
                "cython_time": round(result.cython_time, 6),
                "python_time": round(result.python_time, 6),
                "errors": result.error,
            },
        )

    def get_composite_score(self) -> float:
        """Compute final reward from the last submitted code."""
        if not self.last_code or not self._python_func:
            return 0.0

        scores = composite_reward(
            cython_code=self.last_code,
            python_func=self._python_func,
            func_name=self._func_name,
            test_cases=self._test_cases,
            benchmark_args=self._benchmark_args,
            benchmark_runs=3,
        )
        self.last_scores = scores
        return scores["total"]
