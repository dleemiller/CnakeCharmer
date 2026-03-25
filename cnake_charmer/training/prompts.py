"""
Prompt formatting for multi-turn Cython training.

Structures the system prompt, user turn, and tool definitions
for the model during training rollouts.
"""

SYSTEM_PROMPT = """\
You are an expert Cython developer. Your task is to translate Python code into \
optimized Cython (.pyx) that compiles correctly and runs faster than the original.

You have access to tools that help you verify and improve your code:
- **compile**: Check if your Cython code compiles without errors.
- **annotate**: Analyze your code's optimization quality. Returns a score (0-1) and \
hints about lines that fall back to Python.
- **test**: Run correctness tests comparing your Cython output against the Python reference.
- **benchmark**: Measure the speedup of your Cython code vs the Python original.

Guidelines for good Cython:
- Use `cdef` for C-level variable declarations (int, double, etc.)
- Use `cdef`/`cpdef` for functions that don't need Python-level access
- Use typed memoryviews for array operations instead of numpy indexing
- Add `nogil` where possible to release the GIL
- Use `prange` from cython.parallel for parallelizable loops
- Set compiler directives: boundscheck=False, wraparound=False, cdivision=True
- Minimize PyObject interactions (the yellow lines in annotations)

Output ONLY the complete .pyx file content. Do not include explanations outside the code."""


def format_user_prompt(python_code: str, description: str = "") -> str:
    """Format the user turn asking for a Cython translation."""
    parts = ["Translate this Python code to optimized Cython:"]
    if description:
        parts.append(f"\nDescription: {description}")
    parts.append(f"\n```python\n{python_code}\n```")
    return "\n".join(parts)


def format_feedback(tool_name: str, tool_result: dict) -> str:
    """Format tool result as human-readable feedback for the model."""
    if tool_name == "compile":
        if tool_result["success"]:
            return "Compilation successful."
        return f"Compilation failed:\n{tool_result['errors']}"

    elif tool_name == "annotate":
        if not tool_result["success"]:
            return f"Annotation failed: {tool_result.get('errors', 'unknown error')}"
        score = tool_result["score"]
        hints = tool_result.get("hints", [])
        lines = [
            f"Annotation score: {score:.2f} ({tool_result['yellow_lines']} Python-fallback lines / {tool_result['total_lines']} total)"
        ]
        if hints:
            lines.append("Optimization hints:")
            for h in hints[:5]:
                lines.append(f"  - {h}")
        return "\n".join(lines)

    elif tool_name == "test":
        if not tool_result["success"]:
            return f"Test execution failed: {tool_result.get('errors', 'unknown error')}"
        passed = tool_result["passed"]
        total = tool_result["total"]
        lines = [f"Tests: {passed}/{total} passed"]
        for f in tool_result.get("failures", [])[:5]:
            lines.append(f"  FAIL: {f}")
        return "\n".join(lines)

    elif tool_name == "benchmark":
        if not tool_result["success"]:
            return f"Benchmark failed: {tool_result.get('errors', 'unknown error')}"
        return f"Speedup: {tool_result['speedup']:.2f}x (Python: {tool_result['python_time']:.6f}s, Cython: {tool_result['cython_time']:.6f}s)"

    return str(tool_result)


def make_initial_messages(python_code: str, description: str = "") -> list:
    """Create the initial message list for a training rollout."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_prompt(python_code, description)},
    ]
