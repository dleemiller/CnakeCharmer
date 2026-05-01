"""Monte Carlo Tree Search node utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, log, sqrt
from typing import Any


@dataclass
class Node:
    state: Any
    discovery_factor: float = 0.35
    win_value: float = 0.0
    visits: int = 0
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)

    def update_win_value(self, value: float) -> None:
        self.win_value += value
        self.visits += 1
        if self.parent is not None:
            self.parent.update_win_value(value)

    def add_child(self, child: Node) -> None:
        self.children.append(child)
        child.parent = self

    def add_children(self, children: list[Node]) -> None:
        for child in children:
            self.add_child(child)

    def get_score(self) -> float:
        if self.visits == 0:
            return inf
        if self.parent is None or self.parent.visits <= 0:
            return self.win_value / self.visits
        discovery = self.discovery_factor * sqrt(log(self.parent.visits) / self.visits)
        return self.win_value / self.visits + discovery

    def get_preferred_child(self) -> Node:
        if not self.children:
            raise IndexError("node has no children")
        return max(self.children, key=lambda child: child.get_score())
