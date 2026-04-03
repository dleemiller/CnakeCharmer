"""
Prompt formatting for multi-turn Cython training.

Matches the SFT training data format exactly:
  - System message: data/system_prompt.txt (rendered as Harmony developer message)
  - User message: key-value format (python_code, func_name, description)
  - Tool schemas: loaded from data/tools.json
"""

from pathlib import Path

_SYSTEM_PROMPT_FILE = Path("data/system_prompt.txt")
_TOOLS_FILE = Path("data/tools.json")

# Lazy-loaded system prompt from data/system_prompt.txt
_system_prompt_cache: str | None = None
_tools_cache: list | None = None


def get_system_prompt() -> str:
    """Load the system prompt from data/system_prompt.txt.

    This is the exact prompt used during SFT training, rendered as the
    Harmony developer message by the chat template.
    """
    global _system_prompt_cache
    if _system_prompt_cache is None:
        if _SYSTEM_PROMPT_FILE.exists():
            _system_prompt_cache = _SYSTEM_PROMPT_FILE.read_text().strip()
        else:
            # Fallback if file not found (e.g. running from different directory)
            _system_prompt_cache = (
                "You are a Cython optimization expert. Convert Python code into "
                "fast, correct Cython (.pyx) code."
            )
    return _system_prompt_cache


def get_tools() -> list:
    """Load tool schemas from data/tools.json.

    Returns the exact tool definitions used during SFT training.
    """
    global _tools_cache
    if _tools_cache is None:
        import json

        _tools_cache = json.loads(_TOOLS_FILE.read_text()) if _TOOLS_FILE.exists() else []
    return _tools_cache


def format_user_prompt(python_code: str, func_name: str = "", description: str = "") -> str:
    """Format the user turn in SFT-matching key-value format.

    SFT training data uses:
        python_code: <code>
        func_name: <name>
        description: <desc>
    """
    parts = [f"python_code: {python_code}"]
    if func_name:
        parts.append(f"func_name: {func_name}")
    if description:
        parts.append(f"description: {description}")
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
