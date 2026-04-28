"""Fenwick / binary probability tree for weighted sampling."""

from __future__ import annotations

import random


class FPTree:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be > 0")
        self.size = int(size)
        t_pos = 1
        while t_pos < self.size:
            t_pos *= 2
        self.t_pos = t_pos
        self.values = [0.0] * (2 * t_pos)

    def reset(self) -> None:
        for i in range(1, len(self.values)):
            self.values[i] = 0.0

    def get_value(self, i: int) -> float:
        return self.values[i + self.t_pos]

    def set_value(self, i: int, value: float) -> None:
        if i < 0 or i >= self.size:
            raise IndexError(i)
        if value < 0.0:
            value = 0.0
        pos = i + self.t_pos
        delta = value - self.values[pos]
        while pos > 0:
            self.values[pos] += delta
            pos >>= 1

    def get_total(self) -> float:
        return self.values[1]

    def sample(self, urnd: float) -> int:
        total = self.get_total()
        if total <= 0.0:
            raise ValueError("cannot sample from zero-total tree")
        if urnd < 0.0:
            urnd = 0.0
        if urnd >= total:
            urnd = max(0.0, total - 1e-15)

        pos = 1
        while pos < self.t_pos:
            left = pos * 2
            if urnd >= self.values[left]:
                urnd -= self.values[left]
                pos = left + 1
            else:
                pos = left
        idx = pos - self.t_pos
        return idx if idx < self.size else self.size - 1

    def random_sample(self) -> int:
        return self.sample(random.random() * self.get_total())
