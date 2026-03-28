"""
Data types for SFT trajectory recording and storage.

A Trajectory captures a full multi-turn conversation where a model
uses tools (compile, annotate, test, benchmark) to optimize Python
code to Cython. Used for SFT warmup training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ToolCall:
    """A single tool invocation within a trajectory."""

    tool_name: str  # "compile" | "annotate" | "test" | "benchmark"
    arguments: dict  # {"code": "..."}
    result: str  # tool output string


@dataclass
class Trajectory:
    """A complete multi-turn tool-calling conversation for one problem.

    Stores the full message history in OpenAI chat format (compatible
    with both local vLLM and OpenRouter APIs).
    """

    problem_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    messages: list[dict] = field(default_factory=list)
    final_code: str | None = None
    reward: float = 0.0
    metrics: dict = field(default_factory=dict)
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        """Number of assistant turns in the conversation."""
        return sum(1 for m in self.messages if m.get("role") == "assistant")

    @property
    def num_tool_calls(self) -> int:
        """Total number of tool calls across all turns."""
        count = 0
        for m in self.messages:
            if m.get("tool_calls"):
                count += len(m["tool_calls"])
        return count

    @property
    def tools_used(self) -> set[str]:
        """Set of distinct tool names invoked."""
        tools = set()
        for m in self.messages:
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name"):
                    tools.add(fn["name"])
        return tools

    @property
    def compiled(self) -> bool:
        return self.metrics.get("compilation", False)

    @property
    def correct(self) -> bool:
        return self.metrics.get("correctness", 0.0) == 1.0

    @property
    def speedup(self) -> float:
        return self.metrics.get("speedup", 0.0)
