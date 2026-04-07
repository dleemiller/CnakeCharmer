"""
Tool environment for training and serving a Cython code generation agent.

Exposes `evaluate_cython(code, python_code, test_code)` which:
1. Compiles the Cython code (sandboxed)
2. Runs annotation analysis
3. Executes the model's equivalence tests (sandboxed)
4. Benchmarks against the Python reference (sandboxed)

Works identically for SFT training, GRPO training, and production serving.
Output format is preserved for training data compatibility.
"""

import json
import logging
import math
import os
import signal
import types

from cnake_charmer.eval.annotations import parse_annotations
from cnake_charmer.eval.benchmark import run_benchmark as _run_benchmark
from cnake_charmer.eval.compiler import cleanup_build, compile_cython
from cnake_charmer.eval.correctness import _load_module_from_path
from cnake_charmer.eval.pipeline import DEFAULT_WEIGHTS
from cnake_charmer.eval.runners import COMBINED_RUNNER
from cnake_charmer.training.prompts import format_feedback

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


ASSERTION_TIMEOUT = 5  # seconds per test line


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Timed out")


def _exec_as_module(code: str, name: str = "dynamic_module") -> types.ModuleType:
    """Execute code into a module object. Works for any Python code."""
    mod = types.ModuleType(name)
    exec(code, mod.__dict__)  # noqa: S102
    return mod


def _run_test_code(py_mod, cy_mod, test_code: str) -> dict:
    """Run model's test assertions with py and cy modules in namespace.

    Each line is executed with a 5-second timeout. Lines containing `==`
    are treated as assertions (both sides evaluated, compared). Other
    lines are executed as setup (variable assignments, etc.).

    Returns dict with passed, total, failures list.
    """
    namespace = {"py": py_mod, "cy": cy_mod}
    passed = 0
    total = 0
    failures = []

    for line in test_code.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(ASSERTION_TIMEOUT)
        try:
            if "==" in line:
                # Assertion line: evaluate and compare
                total += 1
                result = eval(line, namespace)  # noqa: S307
                if result:
                    passed += 1
                else:
                    # Try to get both sides for the error message
                    parts = line.split("==", 1)
                    try:
                        left = eval(parts[0].strip(), namespace)  # noqa: S307
                        right = eval(parts[1].strip(), namespace)  # noqa: S307
                        failures.append(
                            f"FAIL: {line}\n  left:  {repr(left)[:200]}\n  right: {repr(right)[:200]}"
                        )
                    except Exception:
                        failures.append(f"FAIL: {line}")
            else:
                # Setup line (variable assignment, import, etc.)
                exec(line, namespace)  # noqa: S102
        except _TimeoutError:
            if "==" in line:
                total += 1
                failures.append(f"TIMEOUT: {line} (>{ASSERTION_TIMEOUT}s)")
            else:
                failures.append(f"TIMEOUT (setup): {line}")
        except Exception as e:
            if "==" in line:
                total += 1
                failures.append(f"ERROR: {line}\n  {type(e).__name__}: {e}")
            else:
                failures.append(f"ERROR (setup): {line}\n  {type(e).__name__}: {e}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return {"passed": passed, "total": total, "failures": failures}


def _run_combined_sandboxed(
    python_code: str,
    cython_module_path: str,
    test_code: str,
    do_benchmark: bool = True,
) -> dict:
    """Run tests and benchmark in a single sandboxed subprocess.

    Returns dict with: py_loaded, cy_loaded, tests, benchmark, and any errors.
    On timeout, returns partial results with a timeout indicator.
    """
    from cnake_charmer.eval.sandbox import execute_config, run_python_sandboxed

    module_dir = os.path.dirname(os.path.abspath(cython_module_path))

    config_data = json.dumps(
        {
            "python_code": python_code,
            "cython_module_path": cython_module_path,
            "test_code": test_code,
            "do_benchmark": do_benchmark,
        }
    )

    # Script that writes config, then runs the combined runner
    full_script = (
        f"""\
import os
with open(os.path.join(os.path.dirname(__file__), '_combined_config.json'), 'w') as f:
    f.write({config_data!r})
"""
        + "\n"
        + COMBINED_RUNNER
    )

    sandbox_cfg = execute_config(
        wall_time_limit_s=45,
        cpu_time_limit_s=30,
        extra_ro_binds=(module_dir,),
    )

    result = run_python_sandboxed(full_script, config=sandbox_cfg)

    if result.timed_out:
        # Return what we can — stdout may have partial output
        return {
            "py_loaded": True,
            "cy_loaded": True,
            "tests": {"passed": 0, "total": 0, "failures": ["TIMEOUT: sandbox timed out"]},
            "timed_out": True,
        }

    if result.returncode != 0:
        stderr_preview = result.stderr[:500] if result.stderr else "(no stderr)"
        return {
            "py_loaded": False,
            "cy_loaded": False,
            "sandbox_error": f"Runner crashed (rc={result.returncode}): {stderr_preview}",
        }

    try:
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return {
            "py_loaded": False,
            "sandbox_error": f"Failed to parse runner output: {result.stdout[:500]}",
        }


class CythonToolEnvironment:
    """Environment for training a Cython code generation agent.

    Exposes `evaluate_cython(code, python_code, test_code)` as the single tool.
    The model provides all three: Cython code, Python reference, and test assertions.

    Used with TRL GRPOTrainer's environment_factory parameter.
    """

    def reset(
        self,
        *,
        python_code: str = "",
        func_name: str = "",
        test_cases: str | list = "[]",
        benchmark_args: str | tuple | None = "null",
        **kwargs,
    ) -> str | None:
        """Reset the environment for a new episode.

        Accepts both the new-style (just python_code for TRL/GRPO) and
        old-style (python_code + func_name + test_cases + benchmark_args
        for DSPy trace collection) kwargs.
        """
        import json as _json

        self.step_scores = []
        self.num_tool_calls = 0
        self.last_code = None
        self._original_python = python_code

        # Old-style: store for legacy evaluate() used by DSPy agent
        self._func_name = func_name
        self._python_func = _exec_func(python_code, func_name) if func_name else None
        self._test_cases = _json.loads(test_cases) if isinstance(test_cases, str) else test_cases
        self._benchmark_args = (
            _json.loads(benchmark_args) if isinstance(benchmark_args, str) else benchmark_args
        )
        if self._benchmark_args is not None and not isinstance(self._benchmark_args, tuple):
            self._benchmark_args = tuple(self._benchmark_args) if self._benchmark_args else None

        return None

    def evaluate_cython(self, code: str, python_code: str, test_code: str) -> str:
        """Compile Cython code, test equivalence against Python reference, and benchmark.

        The test_code runs in a sandboxed namespace where `py` is the Python
        module and `cy` is the compiled Cython module. Each test assertion has
        a 5-second timeout.

        All code execution (tests, benchmark) runs in a bwrap sandbox with
        filesystem/network isolation and resource limits.

        Args:
            code: Complete .pyx Cython source code.
            python_code: Original Python source code (reference implementation).
            test_code: Equivalence test assertions comparing py.<name>(...) == cy.<name>(...).

        Returns:
            Evaluation results as formatted text (identical format to training data).
        """
        # During training, use the ground truth Python from reset() to prevent
        # reward hacking (model could modify python_code to match broken Cython).
        # In production (no reset), use whatever the model passes.
        effective_python = self._original_python if self._original_python else python_code

        self.last_code = code
        self.num_tool_calls += 1
        code_preview = code[:80].replace("\n", "\\n") if code else "<empty>"
        logger.info(
            f"[evaluate_cython] call #{self.num_tool_calls} ({len(code)} chars): {code_preview}..."
        )
        sections = []

        # Step scores
        step = {
            "compiled": False,
            "correctness": 0.0,
            "performance": 0.0,
            "annotations": 0.0,
            "lint": 0.0,
            "memory_safety": 1.0,
            "speedup": 0.0,
            "total": 0.0,
        }

        # 1. Compile (sandboxed via compiler.py)
        comp = compile_cython(code, annotate=True, keep_build=True)
        step["compiled"] = comp.success
        logger.info(f"[evaluate_cython] compiled={comp.success}")
        sections.append(
            "## Compilation\n"
            + format_feedback("compile", {"success": comp.success, "errors": comp.errors})
        )

        if not comp.success:
            self.step_scores.append(step)
            cleanup_build(comp)
            return "\n\n".join(sections)

        # 2. Annotation (safe — just HTML parsing)
        ann = parse_annotations(html_path=comp.html_path) if comp.html_path else None
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

        # 3. Run tests + benchmark in sandboxed subprocess
        if comp.module_path:
            combined = _run_combined_sandboxed(
                python_code=effective_python,
                cython_module_path=comp.module_path,
                test_code=test_code,
                do_benchmark=True,
            )

            # Handle module load errors
            if "py_error" in combined:
                sections.append(f"## Python Error\n{combined['py_error']}")
                self.step_scores.append(step)
                cleanup_build(comp)
                return "\n\n".join(sections)

            if "cy_error" in combined:
                sections.append(f"## Load Error\n{combined['cy_error']}")
                self.step_scores.append(step)
                cleanup_build(comp)
                return "\n\n".join(sections)

            if "sandbox_error" in combined:
                sections.append(f"## Error\n{combined['sandbox_error']}")
                self.step_scores.append(step)
                cleanup_build(comp)
                return "\n\n".join(sections)

            # Process test results
            test_results = combined.get("tests", {"passed": 0, "total": 0, "failures": []})
            step["correctness"] = (
                test_results["passed"] / test_results["total"] if test_results["total"] > 0 else 0.0
            )
            logger.info(f"[evaluate_cython] tests={test_results['passed']}/{test_results['total']}")
            sections.append(
                "## Tests\n"
                + format_feedback(
                    "test",
                    {
                        "success": True,
                        "passed": test_results["passed"],
                        "total": test_results["total"],
                        "failures": test_results["failures"],
                    },
                )
            )

            # Process benchmark results
            bench = combined.get("benchmark")
            if bench:
                if bench.get("success") and bench.get("speedup", 0) > 1.0:
                    step["speedup"] = bench["speedup"]
                    step["performance"] = min(math.log2(bench["speedup"]) / math.log2(10), 1.0)
                logger.info(f"[evaluate_cython] speedup={bench.get('speedup', 0):.1f}x")
                sections.append("## Benchmark\n" + format_feedback("benchmark", bench))

            # Note timeout if it occurred
            if combined.get("timed_out"):
                sections.append(
                    "## Timeout\nThe evaluation timed out. "
                    "This may indicate the function is too slow or has an infinite loop."
                )

        cleanup_build(comp)

        # Compute step total
        step["total"] = self._weighted_score(step)
        self.step_scores.append(step)
        logger.info(
            f"[evaluate_cython] step score: total={step['total']:.3f} "
            f"(correct={step['correctness']:.2f}, perf={step['performance']:.2f}, "
            f"ann={step['annotations']:.2f}, speedup={step['speedup']:.1f}x)"
        )

        return "\n\n".join(sections)

    def evaluate(
        self, code: str, annotate: bool = True, test: bool = True, benchmark: bool = True
    ) -> str:
        """Legacy interface for DSPy trace collection.

        Uses pre-loaded python_code/func_name/test_cases from reset().
        The DSPy agent calls this with just the Cython code; the Python
        reference and test cases come from the dataset via reset().
        """
        from cnake_charmer.eval.correctness import check_correctness

        self.last_code = code
        sections = []

        comp = compile_cython(code, annotate=annotate, keep_build=True)
        sections.append(
            "## Compilation\n"
            + format_feedback("compile", {"success": comp.success, "errors": comp.errors})
        )
        if not comp.success:
            cleanup_build(comp)
            return "\n\n".join(sections)

        # Annotation
        if annotate:
            ann = parse_annotations(html_path=comp.html_path) if comp.html_path else None
            if ann and ann.success:
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

        # Load compiled module
        cython_func = None
        if comp.module_path:
            try:
                module = _load_module_from_path(comp.module_path, "gen_module")
                cython_func = getattr(module, self._func_name)
            except Exception as e:
                sections.append(f"## Load Error\nCould not load function: {e}")
                cleanup_build(comp)
                return "\n\n".join(sections)

        # Test
        if test and cython_func and self._python_func and self._test_cases:
            test_result = check_correctness(
                python_func=self._python_func,
                cython_func=cython_func,
                test_cases=self._test_cases,
            )
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

        # Benchmark (skip if tests failed)
        if (
            benchmark
            and cython_func
            and self._python_func
            and self._test_cases
            and test_result.passed == test_result.total
            and test_result.total > 0
        ):
            b_args = self._benchmark_args or ()
            bench_result = _run_benchmark(
                python_func=self._python_func,
                cython_func=cython_func,
                args=b_args,
                num_runs=3,
            )
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

        cleanup_build(comp)
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
        """R_atomic: Average quality across all tool calls."""
        if not self.step_scores:
            return 0.0
        return sum(self._weighted_score(s, weights) for s in self.step_scores) / len(
            self.step_scores
        )

    def _get_progress_reward(self) -> float:
        """R_progress: Average step-over-step improvement (delta-clipped)."""
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
        if last.get("compiled") and last.get("correctness", 0) >= 1.0:
            bonus += 0.05
        if self.num_tool_calls <= 3 and last.get("compiled"):
            bonus += 0.05
        return bonus
