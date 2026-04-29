"""Descending ordered unique integer list with insertion/removal and shuffled iteration."""

from __future__ import annotations

import random


class OrderedUniqueList:
    def __init__(self) -> None:
        self._vals: list[int] = []

    def add(self, val: int) -> None:
        if val in self._vals:
            return
        i = 0
        while i < len(self._vals) and self._vals[i] > val:
            i += 1
        self._vals.insert(i, val)

    def remove(self, val: int) -> None:
        if val in self._vals:
            self._vals.remove(val)

    def extend(self, additional: OrderedUniqueList) -> None:
        for v in additional._vals:
            self.add(v)

    def clear(self) -> None:
        self._vals.clear()

    def values(self):
        yield from self._vals

    def get(self, index: int) -> int:
        return self._vals[index]

    def get_first(self) -> int:
        return self._vals[0]

    def length(self) -> int:
        return len(self._vals)

    def shuffled(self):
        idx = list(range(len(self._vals)))
        random.shuffle(idx)
        for i in idx:
            yield self._vals[i]
