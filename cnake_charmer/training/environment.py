"""
OpenEnv-compatible tool environment for TRL GRPOTrainer.

Exposes a single `evaluate_cython` tool matching the SFT training data format.
The trainer handles the multi-turn loop: generate → tool call → execute → repeat.

Tracks per-step scores for graduated reward computation (R_atomic + R_progress).
"""

import json
import logging
import math
import multiprocessing

from cnake_charmer.rewards.composite import DEFAULT_WEIGHTS, composite_reward
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


def _evaluate_worker(queue, env_args, code, annotate, test, benchmark):
    """Worker for safe_evaluate — runs in a spawned subprocess."""
    try:
        env = CythonToolEnvironment()
        env.reset(**env_args)
        result = env.evaluate_cython(code=code)
        queue.put(result)
    except Exception as e:
        queue.put(f"## Error\n{type(e).__name__}: {e}")


class CythonToolEnvironment:
    """Environment for training a Cython code generation agent.

    Provides a single `evaluate_cython` tool matching the SFT training format.
    Tracks per-step scores for graduated reward computation.

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

        # Per-step tracking for graduated rewards
        self.step_scores = []  # List of score dicts per tool call
        self.num_tool_calls = 0
        self.format_errors = 0
        self.last_code = None

        return None

    def evaluate_cython(
        self,
        code: str,
        annotate: bool = True,
        test: bool = True,
        benchmark: bool = True,
    ) -> str:
        """Compile, analyze, test, and benchmark Cython code in one step. Returns compilation status, annotation score with optimization hints, correctness test results, and speedup measurement. If compilation fails, only error messages are returned. Fix any issues and call again. Set annotate/test/benchmark to False to skip expensive steps.

        Args:
            code: Complete .pyx source code.
            annotate: Run annotation analysis (default true).
            test: Run correctness tests (default true).
            benchmark: Measure speedup (default true).

        Returns:
            Evaluation results as formatted text.
        """
        self.last_code = code
        self.num_tool_calls += 1
        code_preview = code[:80].replace("\n", "\\n") if code else "<empty>"
        logger.info(
            f"[evaluate_cython] call #{self.num_tool_calls} for {self._func_name} "
            f"({len(code)} chars): {code_preview}..."
        )
        sections = []

        # Compile with annotations enabled (one compilation for everything)
        result = compile_cython(code, annotate=True, keep_build=True)
        compiled = result.success
        logger.info(f"[evaluate_cython] compiled={compiled}")
        sections.append(
            "## Compilation\n"
            + format_feedback("compile", {"success": result.success, "errors": result.errors})
        )

        # Initialize step scores
        step = {
            "compiled": compiled,
            "correctness": 0.0,
            "performance": 0.0,
            "annotations": 0.0,
            "lint": 0.0,
            "memory_safety": 1.0,
            "speedup": 0.0,
            "total": 0.0,
        }

        if not compiled:
            self.step_scores.append(step)
            cleanup_build(result)
            return "\n\n".join(sections)

        # Annotation (from the same compilation, no rebuild)
        ann = parse_annotations(html_path=result.html_path) if result.html_path else None
        if ann and ann.success:
            step["annotations"] = ann.score
            sections.append(
                "## Annotation\n"
                + format_feedback(
                    "annotate",
                    {
                        "success": True,
                        "score": ann.score,
                        "hints": ann.hints,
                        "yellow_lines": ann.yellow_lines,
                        "total_lines": ann.total_lines,
                    },
                )
            )

        # Load the compiled module once for both test and benchmark
        cython_func = None
        if result.module_path:
            try:
                module = _load_module_from_path(result.module_path, "gen_module")
                cython_func = getattr(module, self._func_name)
            except Exception as e:
                sections.append(f"## Load Error\nCould not load function: {e}")
                self.step_scores.append(step)
                cleanup_build(result)
                return "\n\n".join(sections)

        # Test (reuses loaded module)
        if cython_func and self._python_func and self._test_cases:
            test_result = check_correctness(
                python_func=self._python_func,
                cython_func=cython_func,
                test_cases=self._test_cases,
            )
            step["correctness"] = test_result.score
            logger.info(f"[evaluate_cython] tests={test_result.passed}/{test_result.total}")
            sections.append(
                "## Tests\n"
                + format_feedback(
                    "test",
                    {
                        "success": True,
                        "passed": test_result.passed,
                        "total": test_result.total,
                        "failures": test_result.failures,
                    },
                )
            )

        # Benchmark (reuses loaded module) — skip if tests failed to avoid hanging on buggy code
        if cython_func and self._python_func and step["correctness"] > 0:
            b_args = self._benchmark_args or ()
            bench_result = _run_benchmark(
                python_func=self._python_func,
                cython_func=cython_func,
                args=b_args,
                num_runs=3,
            )
            if bench_result.success and bench_result.speedup > 1.0:
                step["speedup"] = bench_result.speedup
                step["performance"] = min(math.log2(bench_result.speedup) / math.log2(10), 1.0)
            logger.info(f"[evaluate_cython] speedup={bench_result.speedup:.1f}x")
            sections.append(
                "## Benchmark\n"
                + format_feedback(
                    "benchmark",
                    {
                        "success": bench_result.success,
                        "speedup": round(bench_result.speedup, 2),
                        "cython_time": round(bench_result.cython_time, 6),
                        "python_time": round(bench_result.python_time, 6),
                        "errors": bench_result.error,
                    },
                )
            )

        cleanup_build(result)

        # Compute weighted total for this step
        step["total"] = self._weighted_score(step)
        self.step_scores.append(step)
        logger.info(
            f"[evaluate_cython] step score: total={step['total']:.3f} "
            f"(correct={step['correctness']:.2f}, perf={step['performance']:.2f}, "
            f"ann={step['annotations']:.2f}, speedup={step['speedup']:.1f}x)"
        )

        return "\n\n".join(sections)

    def _weighted_score(self, scores: dict, weights: dict | None = None) -> float:
        """Compute weighted composite score from a step's scores dict."""
        w = weights or DEFAULT_WEIGHTS
        if not scores.get("compiled"):
            return 0.0
        return (
            w.get("correctness", 0.30) * scores["correctness"]
            + w.get("performance", 0.25) * scores["performance"]
            + w.get("annotations", 0.20) * scores["annotations"]
            + w.get("lint", 0.10) * scores["lint"]
            + w.get("memory_safety", 0.15) * scores["memory_safety"]
        )

    # -- Graduated reward methods (called by reward function) --

    def _get_atomic_reward(self, weights: dict | None = None) -> float:
        """R_atomic: Average quality across all tool calls.

        Every call contributes signal — not just the last one.
        """
        if not self.step_scores:
            return 0.0
        return sum(self._weighted_score(s, weights) for s in self.step_scores) / len(
            self.step_scores
        )

    def _get_progress_reward(self) -> float:
        """R_progress: Average step-over-step improvement (delta-clipped).

        Measures whether each tool call improves on the previous one.
        Asymmetric clipping: improvements rewarded up to +0.5, regressions
        penalized up to -0.3 (encourage exploration).
        """
        if len(self.step_scores) < 2:
            return 0.0
        deltas = []
        for i in range(1, len(self.step_scores)):
            prev = self.step_scores[i - 1]["total"]
            curr = self.step_scores[i]["total"]
            delta = max(-0.3, min(0.5, curr - prev))
            deltas.append(delta)
        return sum(deltas) / len(deltas)

    def _get_bonus_reward(self) -> float:
        """R_bonus: Small bonuses/penalties for desirable behaviors."""
        bonus = 0.0
        if not self.step_scores:
            return bonus

        last = self.step_scores[-1]
        # Completion bonus: final code compiles and passes all tests
        if last.get("compiled") and last.get("correctness", 0) >= 1.0:
            bonus += 0.05
        # Efficiency bonus: solved in ≤3 tool calls
        if self.num_tool_calls <= 3 and last.get("compiled"):
            bonus += 0.05
        # Format penalty
        if self.format_errors > 0:
            bonus -= 0.1 * self.format_errors
        return bonus

    def _get_composite_score(self) -> float:
        """Legacy: compute final reward from the last submitted code.

        Kept for backward compatibility with existing code.
        """
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
        return scores["total"]

    def _safe_evaluate(
        self,
        code: str,
        timeout: int = 60,
    ) -> str:
        """Run evaluate_cython() in a subprocess to isolate segfaults and hangs.

        Args:
            code: Complete .pyx source code.
            timeout: Max seconds before killing the worker.

        Returns:
            Markdown-formatted results, or error message on crash/timeout.
        """
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        env_args = {
            "python_code": self._python_code,
            "func_name": self._func_name,
            "test_cases": self._test_cases,
            "benchmark_args": self._benchmark_args,
        }

        proc = ctx.Process(
            target=_evaluate_worker,
            args=(queue, env_args, code, True, True, True),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.kill()
            proc.join()
            return "## Error\nEvaluation timed out."

        if proc.exitcode != 0:
            return f"## Error\nEvaluation crashed (exit code {proc.exitcode}). Try a different approach."

        if not queue.empty():
            return queue.get()

        return "## Error\nEvaluation returned no result."
