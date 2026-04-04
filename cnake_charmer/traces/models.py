"""
Pydantic models for the standardized trace format.

Traces capture a single agent attempt at optimizing one problem.
The v2 format uses structured ToolStep lists instead of flat
tool_name_0/tool_args_0/observation_0 keys.

Key distinction preserved from v1:
  - thought_N: DSPy ReAct chain-of-thought (brief planning per step)
  - reasoning_N: Model reasoning_content from thinking API (deep reasoning)
  These are separate channels indexed by step. reasoning_N may extend
  beyond the tool steps (extra reasoning after the last tool call).
"""

from __future__ import annotations

import contextlib
from datetime import datetime

from pydantic import BaseModel


class ToolStep(BaseModel):
    """One tool call and its result within a trace."""

    tool_name: str
    tool_args: dict
    observation: str = ""
    thought: str | None = None  # DSPy ReAct thought (brief planning)
    reasoning: str | None = None  # Model reasoning_content (thinking API)


class Trace(BaseModel):
    """Single agent trace: one problem, one attempt."""

    problem_id: str
    model: str
    prompt_id: str = ""
    attempt: int = 0
    timestamp: datetime | None = None
    steps: list[ToolStep] = []
    # Extra reasoning entries beyond the last tool step (reasoning_N where N >= len(steps))
    trailing_reasoning: list[str] = []
    final_code: str | None = None
    reward: float = 0.0
    metrics: dict = {}
    thinking: bool = False
    func_name: str = ""
    category: str = ""
    difficulty: str = ""
    version: str = "2.0"

    @property
    def num_iterations(self) -> int:
        return len(self.steps)

    @property
    def tools_used(self) -> list[str]:
        return list({s.tool_name for s in self.steps})

    def to_flat_trajectory(self) -> dict:
        """Convert structured steps back to flat trajectory dict (v1 compat).

        Faithfully reconstructs thought_N, reasoning_N, tool_name_N,
        tool_args_N, observation_N keys including trailing reasoning.
        """
        traj = {}
        for i, step in enumerate(self.steps):
            if step.thought:
                traj[f"thought_{i}"] = step.thought
            if step.reasoning:
                traj[f"reasoning_{i}"] = step.reasoning
            traj[f"tool_name_{i}"] = step.tool_name
            traj[f"tool_args_{i}"] = step.tool_args
            traj[f"observation_{i}"] = step.observation
        # Trailing reasoning (indexed continuing from len(steps))
        for j, r in enumerate(self.trailing_reasoning):
            traj[f"reasoning_{len(self.steps) + j}"] = r
        return traj

    def to_v1_dict(self) -> dict:
        """Export to v1-compatible flat dict for backward compat."""
        return {
            "version": "1.0",
            "model": self.model,
            "prompt_id": self.prompt_id,
            "problem_id": self.problem_id,
            "func_name": self.func_name,
            "category": self.category,
            "difficulty": self.difficulty,
            "attempt": self.attempt,
            "num_iterations": self.num_iterations,
            "tools_used": self.tools_used,
            "trajectory": self.to_flat_trajectory(),
            "cython_code": self.final_code or "",
            "output": None,
            "reward": self.reward,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_v1_dict(cls, d: dict) -> Trace:
        """Parse a v1 flat-key trace dict into a structured Trace.

        Preserves thought_N and reasoning_N as separate channels.
        Handles reasoning_N entries beyond tool steps as trailing_reasoning.
        """
        traj = d.get("trajectory", {})

        # Parse tool steps
        steps = []
        i = 0
        while f"tool_name_{i}" in traj:
            tool_name = traj[f"tool_name_{i}"]
            if tool_name is not None:
                steps.append(
                    ToolStep(
                        tool_name=str(tool_name),
                        tool_args=_parse_tool_args(traj.get(f"tool_args_{i}", {})),
                        observation=str(traj.get(f"observation_{i}", "")),
                        thought=traj.get(f"thought_{i}"),
                        reasoning=traj.get(f"reasoning_{i}"),
                    )
                )
            i += 1

        # Collect trailing reasoning entries (reasoning_N where N >= num tool steps)
        n_steps = len(steps)
        trailing = []
        j = n_steps
        while f"reasoning_{j}" in traj:
            trailing.append(traj[f"reasoning_{j}"])
            j += 1

        # Detect thinking: reasoning_N keys in trajectory
        thinking = any(k.startswith("reasoning_") for k in traj)

        ts = None
        if d.get("timestamp"):
            with contextlib.suppress(ValueError, TypeError):
                ts = datetime.fromisoformat(d["timestamp"])

        return cls(
            problem_id=d.get("problem_id", ""),
            model=d.get("model", ""),
            prompt_id=d.get("prompt_id", ""),
            attempt=d.get("attempt", 0),
            timestamp=ts,
            steps=steps,
            trailing_reasoning=trailing,
            final_code=d.get("cython_code", ""),
            reward=d.get("reward") or 0.0,
            metrics=d.get("metrics", {}),
            thinking=thinking,
            func_name=d.get("func_name", ""),
            category=d.get("category", ""),
            difficulty=d.get("difficulty", ""),
            version="2.0",
        )


def _parse_tool_args(args) -> dict:
    """Normalize tool args to dict — handles str or dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        import json

        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"raw": args}
    return {}
