"""
DSPy-based Cython optimization agent.

Defines a ReAct agent with compile/annotate/test/benchmark tools,
a GEPA-compatible metric using our composite reward, and helper
functions for trace collection via dspy-data's Collect.
"""

import json
import logging
import multiprocessing
import re

import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

from cnake_charmer.rewards.composite import composite_reward
from cnake_charmer.training.environment import CythonToolEnvironment, _exec_func
from cnake_charmer.training.rollout import extract_code_from_content

# Imports that should never appear in generated Cython code
_UNSAFE_IMPORTS = {
    "os",
    "shutil",
    "subprocess",
    "sys",
    "pathlib",
    "socket",
    "http",
    "urllib",
    "requests",
    "pickle",
    "shelve",
    "tempfile",
    "glob",
}


def _check_code_safety(code: str) -> str | None:
    """Check generated code for unsafe imports. Returns error message or None if safe."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for mod in _UNSAFE_IMPORTS:
            if re.search(rf"\bimport\s+{mod}\b", stripped) or re.search(
                rf"\bfrom\s+{mod}\b", stripped
            ):
                return f"Unsafe import detected: '{mod}'. Generated code must not use {mod}."
    return None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------


class CythonOptimization(dspy.Signature):
    """Translate Python code into optimized Cython (.pyx) that compiles correctly and runs faster.
    You MUST use all four tools in order: compile_cython → annotate_cython → test_cython → benchmark_cython.
    Do not finish until you have verified compilation, checked annotation quality, confirmed all tests pass, and measured the speedup."""

    python_code: str = dspy.InputField(desc="The Python source code to optimize")
    func_name: str = dspy.InputField(desc="Name of the main function to optimize")
    description: str = dspy.InputField(desc="Description of what the code does", default="")

    cython_code: str = dspy.OutputField(desc="Complete .pyx Cython source code")


# ---------------------------------------------------------------------------
# Tool wrappers (stateful — share a CythonToolEnvironment per problem)
# ---------------------------------------------------------------------------


def _tool_worker(queue, tool_name, env_args, code):
    """Worker for sandboxed tool calls. Runs in a spawn subprocess."""
    try:
        env = CythonToolEnvironment()
        env.reset(**env_args)
        result = getattr(env, tool_name)(code)
        queue.put(result)
    except Exception as e:
        queue.put(f"Tool error: {type(e).__name__}: {e}")


def _sandboxed_tool_call(tool_name, env_args, code, timeout=30):
    """Run a tool call in a spawn subprocess to isolate segfaults."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_tool_worker, args=(queue, tool_name, env_args, code))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return f"Tool timed out after {timeout}s."

    if proc.exitcode != 0:
        return (
            f"Tool crashed (exit {proc.exitcode}). The code likely causes a segfault when loaded."
        )

    if not queue.empty():
        return queue.get()

    return "Tool returned no result."


def make_tools(
    python_code: str, func_name: str, test_cases: list, benchmark_args: tuple | None = None
):
    """Create DSPy-compatible tool functions for a specific problem.

    Each tool runs in a spawn subprocess to isolate segfaults from
    compiled Cython modules that may crash when loaded.
    """
    env_args = {
        "python_code": python_code,
        "func_name": func_name,
        "test_cases": json.dumps(test_cases),
        "benchmark_args": json.dumps(benchmark_args),
    }

    # Also create an in-process env for compile (safe — doesn't load the .so)
    env = CythonToolEnvironment()
    env.reset(**env_args)

    def compile_cython(code: str) -> str:
        """Compile Cython (.pyx) code. Call this first after writing or modifying code.
        Returns 'Compilation successful.' or error messages to fix before proceeding.

        Args:
            code: Complete .pyx source code.
        """
        safety_err = _check_code_safety(code)
        if safety_err:
            return safety_err
        return env.compile(code)

    def annotate_cython(code: str) -> str:
        """Analyze optimization quality via Cython's HTML annotations. Returns a score
        (0.0 = all Python fallback, 1.0 = pure C) and hints about which lines need
        cdef declarations or C-library replacements. Call after successful compilation.

        Args:
            code: Complete .pyx source code.
        """
        # Annotate is safe in-process (parses HTML, doesn't load .so)
        return env.annotate(code)

    def test_cython(code: str) -> str:
        """Run correctness tests comparing Cython output against the Python reference.
        All tests must pass. Failures show which inputs produced wrong results.

        Args:
            code: Complete .pyx source code.
        """
        return _sandboxed_tool_call("test", env_args, code)

    def benchmark_cython(code: str) -> str:
        """Measure speedup vs the Python original. Good results are 2x+, excellent is 10x+.
        Reports wall-clock times for both implementations.

        Args:
            code: Complete .pyx source code.
        """
        return _sandboxed_tool_call("benchmark", env_args, code)

    tools = [compile_cython, annotate_cython, test_cython, benchmark_cython]
    return tools, env


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent(max_iters: int = 5) -> dspy.ReAct:
    """Create a ReAct agent for Cython optimization.

    Note: Tools are problem-specific, so this creates the agent structure
    with placeholder tools. Use make_tools() to create actual tools per problem,
    then pass them when calling the agent.
    """

    def _placeholder_compile(code: str) -> str:
        """Compile Cython code and check for errors."""
        return "placeholder"

    def _placeholder_annotate(code: str) -> str:
        """Analyze Cython code optimization quality."""
        return "placeholder"

    def _placeholder_test(code: str) -> str:
        """Run correctness tests against the Python reference."""
        return "placeholder"

    def _placeholder_benchmark(code: str) -> str:
        """Measure speedup of Cython code vs Python."""
        return "placeholder"

    return dspy.ReAct(
        CythonOptimization,
        tools=[
            _placeholder_compile,
            _placeholder_annotate,
            _placeholder_test,
            _placeholder_benchmark,
        ],
        max_iters=max_iters,
    )


# ---------------------------------------------------------------------------
# Sandboxed reward computation (prevents segfaults from crashing the process)
# ---------------------------------------------------------------------------

_ZERO_SCORES = {
    "total": 0.0,
    "compiled": False,
    "correctness": 0.0,
    "performance": 0.0,
    "annotations": 0.0,
    "lint": 0.0,
    "memory_safety": 0.0,
    "speedup": 0.0,
    "compilation_errors": "",
    "correctness_failures": [],
    "annotation_hints": [],
    "lint_violations": [],
    "memory_safety_errors": [],
}


def _reward_worker(queue, cython_code, python_code, func_name, test_cases, benchmark_args):
    """Worker that runs composite_reward in an isolated process."""
    import traceback

    try:
        python_func = _exec_func(python_code, func_name)
        if python_func is None:
            queue.put({**_ZERO_SCORES, "compilation_errors": "Could not exec Python"})
            return
        scores = composite_reward(
            cython_code=cython_code,
            python_func=python_func,
            func_name=func_name,
            test_cases=test_cases,
            benchmark_args=benchmark_args,
            benchmark_runs=3,
        )
        queue.put(scores)
    except Exception as e:
        tb = traceback.format_exc()
        logger.warning(f"Reward worker exception: {tb}")
        queue.put({**_ZERO_SCORES, "compilation_errors": f"{type(e).__name__}: {e}"})


def _safe_composite_reward(
    cython_code: str,
    python_code: str,
    func_name: str,
    test_cases: list,
    benchmark_args: tuple | None = None,
    timeout: int = 30,
) -> dict:
    """Run composite_reward in a subprocess to isolate segfaults.

    Uses 'spawn' start method (not 'fork') to avoid inheriting corrupted
    state from loaded Cython .so files that may segfault.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_reward_worker,
        args=(queue, cython_code, python_code, func_name, test_cases, benchmark_args),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        logger.warning("Reward computation timed out")
        return {**_ZERO_SCORES, "compilation_errors": "Timed out"}

    if proc.exitcode != 0:
        logger.warning(f"Reward worker crashed with exit code {proc.exitcode}")
        return {**_ZERO_SCORES, "compilation_errors": f"Worker crashed (exit {proc.exitcode})"}

    if not queue.empty():
        return queue.get()

    return {**_ZERO_SCORES, "compilation_errors": "No result from worker"}


# ---------------------------------------------------------------------------
# GEPA metric
# ---------------------------------------------------------------------------


def cython_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: object | None = None,
    pred_name: str | None = None,
    pred_trace: object | None = None,
) -> float | ScoreWithFeedback:
    """GEPA-compatible metric using composite reward + textual feedback.

    When called without pred_name (module-level scoring), returns ScoreWithFeedback
    with detailed feedback for reflection. When called with pred_name (predictor-level),
    returns the same score to avoid GEPA's score mismatch warning.
    """
    cython_code = extract_code_from_content(getattr(pred, "cython_code", ""))
    python_code = gold.get("python_code", "")
    func_name = gold.get("func_name", "")
    test_cases = gold.get("test_cases", [])
    benchmark_args = gold.get("benchmark_args", None)

    if not cython_code or not python_code or not func_name:
        feedback = "Missing required fields (cython_code, python_code, or func_name)."
        if pred_name:
            return ScoreWithFeedback(score=0.0, feedback=feedback)
        return ScoreWithFeedback(score=0.0, feedback=feedback)

    python_func = _exec_func(python_code, func_name)
    if python_func is None:
        return ScoreWithFeedback(score=0.0, feedback="Could not execute Python reference code.")

    # Parse test_cases if they're JSON strings
    if isinstance(test_cases, str):
        test_cases = json.loads(test_cases)
    if isinstance(benchmark_args, str):
        benchmark_args = json.loads(benchmark_args)

    scores = _safe_composite_reward(
        cython_code=cython_code,
        python_code=python_code,
        func_name=func_name,
        test_cases=test_cases,
        benchmark_args=benchmark_args,
    )

    total = scores["total"]

    # Build textual feedback for GEPA reflection
    feedback_parts = []

    if not scores["compiled"]:
        feedback_parts.append(
            f"Compilation FAILED: {scores.get('compilation_errors', 'unknown error')}"
        )
        return ScoreWithFeedback(score=0.0, feedback="\n".join(feedback_parts))

    feedback_parts.append(f"Compiled successfully. Total reward: {total:.3f}")
    feedback_parts.append(f"Correctness: {scores['correctness']:.2f} (test pass rate)")
    feedback_parts.append(
        f"Performance: {scores['performance']:.2f} (speedup: {scores['speedup']:.1f}x)"
    )
    feedback_parts.append(
        f"Annotation quality: {scores['annotations']:.2f} (ratio of pure-C lines)"
    )
    feedback_parts.append(f"Lint: {scores['lint']:.2f}")
    feedback_parts.append(f"Memory safety: {scores['memory_safety']:.2f}")

    if scores.get("correctness_failures"):
        feedback_parts.append("Test failures (model produced wrong output for these inputs):")
        for f in scores["correctness_failures"][:3]:
            feedback_parts.append(f"  - {f}")

    if scores.get("annotation_hints"):
        feedback_parts.append("Cython optimization hints (lines falling back to Python):")
        for hint in scores["annotation_hints"][:3]:
            feedback_parts.append(f"  - {hint}")

    if scores.get("lint_violations"):
        feedback_parts.append(f"Lint violations: {scores['lint_violations'][:3]}")

    # Check tool usage from the prediction's trajectory
    pred_trajectory = getattr(pred, "trajectory", None)
    if isinstance(pred_trajectory, dict):
        tools_used = set()
        idx = 0
        while f"tool_name_{idx}" in pred_trajectory:
            tools_used.add(pred_trajectory[f"tool_name_{idx}"])
            idx += 1
        missing_tools = REQUIRED_TOOLS - tools_used
        if missing_tools:
            feedback_parts.append(
                f"CRITICAL: The agent did NOT use these required tools: {', '.join(sorted(missing_tools))}. "
                f"The prompt MUST instruct the agent to use ALL four tools "
                f"(compile → annotate → test → benchmark) before finishing. "
                f"Traces that skip test or benchmark are rejected with score 0."
            )
            # Penalize heavily for skipping tools
            total *= 0.5

    # Actionable summary for the reflection model
    if scores["correctness"] < 1.0:
        feedback_parts.append(
            "ACTION: The generated code has correctness bugs. The prompt should emphasize running test_cython and fixing failures."
        )
    if scores["speedup"] < 2.0:
        feedback_parts.append(
            "ACTION: Speedup is low. The prompt should emphasize running benchmark_cython and optimizing based on results."
        )
    if scores["annotations"] < 0.8:
        feedback_parts.append(
            "ACTION: Many lines fall back to Python. The prompt should stress using cdef, typed memoryviews, and avoiding Python calls in hot loops."
        )

    # Include ground-truth Cython when the model underperformed, so the
    # reflection model can compare and identify what techniques are missing.
    ground_truth = gold.get("cython_code", "")
    if ground_truth and total < 0.85:
        feedback_parts.append(
            f"\nREFERENCE SOLUTION (ground-truth optimized Cython that scores ~1.0):\n"
            f"```cython\n{ground_truth}\n```\n"
            f"Compare the generated code against this reference to identify "
            f"what optimization techniques or patterns the prompt should encourage."
        )

    return ScoreWithFeedback(score=total, feedback="\n".join(feedback_parts))


# ---------------------------------------------------------------------------
# Process reward: credit tool usage patterns (Context-1 insight)
# ---------------------------------------------------------------------------

ALL_TOOLS = {"compile_cython", "annotate_cython", "test_cython", "benchmark_cython"}

# Required tools — trace must use all of these or reward is zeroed
REQUIRED_TOOLS = {"compile_cython", "annotate_cython", "test_cython", "benchmark_cython"}

# Per-tool bonus for using each distinct tool
TOOL_DIVERSITY_BONUS = 0.025  # +0.025 per tool used, max +0.10 for all 4

# Bonus for iterative improvement (compile → fail → fix → compile again)
ITERATION_BONUS = 0.02  # +0.02 per additional compile call beyond the first


def extract_tool_usage(entry: dict) -> dict:
    """Extract tool usage statistics from a collected entry.

    Uses the structured trajectory dict (tool_name_N, tool_args_N, observation_N)
    captured by dspy-data's ScoreAndSaveWrapper. Falls back to text parsing
    of the trace if no trajectory is available.

    Returns:
        Dict with tool_calls (list), tools_used (set), compile_count, etc.
    """
    tool_calls = []
    tools_used = set()

    # Prefer structured trajectory (clean, no parsing needed)
    trajectory = entry.get("trajectory")
    if isinstance(trajectory, dict):
        idx = 0
        while f"tool_name_{idx}" in trajectory:
            tool_name = trajectory[f"tool_name_{idx}"]
            if tool_name in ALL_TOOLS:
                tool_calls.append(tool_name)
                tools_used.add(tool_name)
            idx += 1
    else:
        # Fallback: parse from trace completions
        for interaction in entry.get("trace", []):
            comp = interaction.get("completion") or {}
            for choice in comp.get("choices", []):
                msg = choice.get("message") or {}
                content = msg.get("content", "") or ""
                for match in re.finditer(
                    r"\[\[\s*##\s*next_tool_name\s*##\s*\]\]\s*\n(\w+)", content
                ):
                    tool_name = match.group(1).strip()
                    if tool_name in ALL_TOOLS:
                        tool_calls.append(tool_name)
                        tools_used.add(tool_name)

    compile_count = sum(1 for t in tool_calls if t == "compile_cython")
    return {
        "tool_calls": tool_calls,
        "tools_used": tools_used,
        "total_calls": len(tool_calls),
        "compile_count": compile_count,
        "unique_tools": len(tools_used),
    }


def process_reward(entry: dict, require_all_tools: bool = True) -> float:
    """Compute process reward from tool usage patterns.

    When require_all_tools=True, returns -1.0 if any required tool is missing,
    signaling that this trace should be rejected. Otherwise returns a bonus.

    Returns:
        -1.0 if tool gate failed (trace should be scored as 0),
        or bonus reward (0.0 to ~0.20) to add to outcome reward.
    """
    usage = extract_tool_usage(entry)

    # Gate: must use all 4 tools at least once
    if require_all_tools:
        missing = REQUIRED_TOOLS - usage["tools_used"]
        if missing:
            return -1.0  # signal to zero the total reward

    bonus = 0.0

    # Tool diversity: +0.025 per unique tool used (max +0.10 for all 4)
    bonus += usage["unique_tools"] * TOOL_DIVERSITY_BONUS

    # Iterative improvement: bonus for re-compiling (implies fix-and-retry)
    if usage["compile_count"] > 1:
        extra_compiles = min(usage["compile_count"] - 1, 3)  # cap at 3 extra
        bonus += extra_compiles * ITERATION_BONUS

    return bonus


# ---------------------------------------------------------------------------
# Reward function for dspy-data Collect
# ---------------------------------------------------------------------------


def collect_reward(inputs: dict, prediction: dspy.Prediction) -> float:
    """Reward function compatible with dspy-data's Collect/ScoreAndSaveWrapper.

    Combines outcome reward (composite_reward on final code) with process
    reward (tool usage bonus, per Context-1's approach).
    """
    cython_code = extract_code_from_content(getattr(prediction, "cython_code", ""))
    python_code = inputs.get("python_code", "")
    func_name = inputs.get("func_name", "")
    test_cases = inputs.get("test_cases", [])
    benchmark_args = inputs.get("benchmark_args")

    if not cython_code or not python_code or not func_name:
        return 0.0

    python_func = _exec_func(python_code, func_name)
    if python_func is None:
        return 0.0

    if isinstance(test_cases, str):
        test_cases = json.loads(test_cases)
    if isinstance(benchmark_args, str):
        benchmark_args = json.loads(benchmark_args)

    scores = _safe_composite_reward(
        cython_code=cython_code,
        python_code=python_code,
        func_name=func_name,
        test_cases=test_cases,
        benchmark_args=benchmark_args,
    )

    return scores["total"]


def score_trajectory(entry: dict, require_all_tools: bool = True) -> float:
    """Score a collected trace entry with both outcome and process rewards.

    Use this when loading/filtering traces for SFT data selection.
    Combines the stored outcome reward with process reward from tool usage.
    If require_all_tools=True and any required tool is missing, returns 0.0.
    """
    outcome = entry.get("reward", 0.0) or 0.0
    bonus = process_reward(entry, require_all_tools=require_all_tools)
    if bonus == -1.0:
        return 0.0  # tool gate failed
    return min(outcome + bonus, 1.0)  # cap at 1.0


# ---------------------------------------------------------------------------
# Problem → DSPy Example conversion
# ---------------------------------------------------------------------------


def problem_to_example(problem) -> dspy.Example:
    """Convert a ProblemSpec to a DSPy Example for training/optimization."""
    return dspy.Example(
        python_code=problem.python_code,
        func_name=problem.func_name,
        description=problem.description,
        test_cases=json.dumps(problem.test_cases),
        benchmark_args=json.dumps(problem.benchmark_args),
        cython_code=problem.cython_code,  # ground truth for reference
    ).with_inputs("python_code", "func_name", "description")
