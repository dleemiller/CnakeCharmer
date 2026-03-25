"""
MURPHY-style credit assignment for multi-turn rollouts.

When using ART's built-in GRPO, each trajectory gets a single reward
and GRPO normalizes within groups. This module provides additional
credit assignment strategies from the MURPHY paper for future use:

- MARS (Max Reward Strategy): Each node's credit = max(own reward, best descendant)
- MeRS (Mean Reward Strategy): Each node's credit = own + discounted mean of children

These are useful when building rollout trees (multiple revision paths
per initial generation). With ART's flat trajectory model, these
aren't needed initially but become relevant for MURPHY-tree rollouts.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RolloutNode:
    """A node in a MURPHY rollout tree."""

    code: str
    reward: float = 0.0
    turn: int = 0
    solved: bool = False
    children: list = field(default_factory=list)
    # Credit-assigned reward (set by MARS/MeRS)
    credit: float = 0.0


def mars_credit(node: RolloutNode, gamma: float = 0.9) -> float:
    """
    Max Reward Strategy: propagate max descendant reward upward.

    credit(node) = max(node.reward, max(credit(child) for child in children))

    Emphasizes peak performance — if any descendant solves the problem,
    the ancestor that led to it gets credit.
    """
    if not node.children:
        node.credit = node.reward
        return node.credit

    child_credits = [mars_credit(child, gamma) for child in node.children]
    best_descendant = max(child_credits)

    if node.solved:
        node.credit = node.reward
    else:
        # Depth-normalized: discount by remaining turns
        depth_discount = 1.0  # Could add: / (S - node.turn + 1)
        node.credit = max(node.reward, gamma * best_descendant * depth_discount)

    return node.credit


def mers_credit(node: RolloutNode, gamma: float = 0.9) -> float:
    """
    Mean Reward Strategy: propagate discounted mean of children.

    credit(node) = node.reward + gamma * mean(credit(children))

    Provides smoother signal — rewards consistent improvement
    rather than lucky single paths.
    """
    if not node.children:
        node.credit = node.reward
        return node.credit

    child_credits = [mers_credit(child, gamma) for child in node.children]

    # Only average over unsolved children (solved ones are terminal)
    unsolved_credits = [
        c for c, child in zip(child_credits, node.children, strict=True) if not child.solved
    ]

    if unsolved_credits:
        mean_child = sum(unsolved_credits) / len(unsolved_credits)
        depth_norm = len(node.children)  # Could normalize by (S - turn + 1)
        node.credit = node.reward + gamma * mean_child / max(1, depth_norm)
    else:
        node.credit = node.reward

    return node.credit


def build_rollout_tree(
    trajectories: list,
    rewards: list,
    turns_per_traj: int = 2,
) -> RolloutNode:
    """
    Build a rollout tree from flat trajectories.

    This is a placeholder for when we implement MURPHY-style tree rollouts.
    Currently, ART trajectories are flat (each is independent), so the tree
    is just a root with leaf children.

    In full MURPHY, failed turn-1 generations branch into turn-2 generations,
    forming a tree. This function would reconstruct that structure.
    """
    root = RolloutNode(code="", turn=0)

    for traj, reward in zip(trajectories, rewards, strict=True):
        leaf = RolloutNode(
            code=str(traj),
            reward=reward,
            turn=1,
            solved=reward > 0.8,
        )
        root.children.append(leaf)

    return root
