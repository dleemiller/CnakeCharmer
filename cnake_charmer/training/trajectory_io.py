"""
Trajectory serialization, loading, and conversion to SFT format.

Storage format: JSONL (one trajectory per line) for streaming
and resume-friendly append-only writing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from cnake_charmer.training.trajectory import Trajectory

logger = logging.getLogger(__name__)


def save_trajectories(
    path: str | Path, trajectories: list[Trajectory], *, append: bool = False
) -> int:
    """Save trajectories to a JSONL file.

    Args:
        path: Output file path.
        trajectories: List of trajectories to save.
        append: If True, append to existing file instead of overwriting.

    Returns:
        Number of trajectories written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with open(path, mode) as f:
        for traj in trajectories:
            f.write(json.dumps(asdict(traj), default=str) + "\n")

    return len(trajectories)


def append_trajectory(path: str | Path, trajectory: Trajectory) -> None:
    """Append a single trajectory to a JSONL file (resume-friendly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(trajectory), default=str) + "\n")


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL file.

    Returns:
        List of Trajectory objects. Skips malformed lines with a warning.
    """
    path = Path(path)
    if not path.exists():
        return []

    trajectories = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                trajectories.append(_dict_to_trajectory(data))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Skipping malformed trajectory at line {i}: {e}")

    return trajectories


def _dict_to_trajectory(data: dict) -> Trajectory:
    """Convert a dict (from JSON) back to a Trajectory."""
    return Trajectory(
        problem_id=data["problem_id"],
        difficulty=data.get("difficulty", ""),
        messages=data.get("messages", []),
        final_code=data.get("final_code"),
        reward=data.get("reward", 0.0),
        metrics=data.get("metrics", {}),
        model=data.get("model", ""),
        timestamp=data.get("timestamp", ""),
        metadata=data.get("metadata", {}),
    )


def get_completed_ids(path: str | Path) -> set[str]:
    """Get set of problem_ids already in a trajectory file.

    Useful for resuming interrupted generation runs.
    """
    path = Path(path)
    if not path.exists():
        return set()

    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ids.add(data["problem_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def trajectory_stats(trajectories: list[Trajectory]) -> dict:
    """Compute summary statistics over a collection of trajectories."""
    if not trajectories:
        return {"count": 0}

    rewards = [t.reward for t in trajectories]
    compiled = sum(1 for t in trajectories if t.compiled)
    correct = sum(1 for t in trajectories if t.correct)
    speedups = [t.speedup for t in trajectories if t.speedup > 0]

    return {
        "count": len(trajectories),
        "compiled": compiled,
        "compiled_pct": round(compiled / len(trajectories) * 100, 1),
        "correct": correct,
        "correct_pct": round(correct / len(trajectories) * 100, 1),
        "mean_reward": round(sum(rewards) / len(rewards), 3),
        "max_reward": round(max(rewards), 3),
        "min_reward": round(min(rewards), 3),
        "mean_speedup": round(sum(speedups) / len(speedups), 2) if speedups else 0.0,
        "mean_turns": round(sum(t.num_turns for t in trajectories) / len(trajectories), 1),
        "models": dict(
            sorted(
                {
                    m: sum(1 for t in trajectories if t.model == m)
                    for m in {t.model for t in trajectories}
                }.items()
            )
        ),
        "difficulty": {
            d: sum(1 for t in trajectories if t.difficulty == d)
            for d in ("easy", "medium", "hard")
            if any(t.difficulty == d for t in trajectories)
        },
    }


def to_sft_messages(trajectory: Trajectory) -> list[dict]:
    """Convert a trajectory to the chat message format for SFT training.

    Returns the raw message list from the trajectory, which is already
    in OpenAI chat format with tool_calls. The SFT trainer (TRL)
    handles tokenization via the model's chat template.
    """
    return trajectory.messages
