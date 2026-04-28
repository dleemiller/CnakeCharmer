"""Array-backed min-heap with decrease-key support."""

from __future__ import annotations


def _parent(i: int) -> int:
    return (i - 1) // 2


def _left(i: int) -> int:
    return 2 * i + 1


def _right(i: int) -> int:
    return 2 * i + 2


class MinHeap:
    def __init__(self, n: int):
        self.val: list[int] = []
        self.pos: list[int] = [-1] * n
        self.size = 0

    def empty(self) -> bool:
        return self.size == 0

    def _swap(self, x: int, y: int) -> None:
        self.val[x], self.val[y] = self.val[y], self.val[x]
        self.pos[self.val[x]] = x
        self.pos[self.val[y]] = y

    def insert_key(self, k: int, scores) -> None:
        if self.size == len(self.val):
            self.val.append(k)
        else:
            self.val[self.size] = k
        self.pos[k] = self.size
        i = self.size
        self.size += 1
        p = _parent(i)
        while p >= 0 and scores[self.val[p]] > scores[self.val[i]]:
            self._swap(i, p)
            i = p
            p = _parent(i)

    def decrease_key(self, i: int, scores) -> None:
        pos = self.pos[i]
        if pos < self.size:
            p = _parent(pos)
            while pos != 0 and scores[self.val[p]] > scores[self.val[pos]]:
                self._swap(pos, p)
                pos = p
                p = _parent(pos)

    def pop_min(self, scores) -> int:
        if self.size <= 0:
            raise IndexError("pop from empty heap")
        if self.size == 1:
            self.size = 0
            return self.val[0]
        root = self.val[0]
        self.val[0] = self.val[self.size - 1]
        self.pos[self.val[0]] = 0
        self.size -= 1
        self.min_heapify(0, scores)
        return root

    def min_heapify(self, i: int, scores) -> None:
        left_idx = _left(i)
        r = _right(i)
        smallest = i
        if left_idx < self.size and scores[self.val[left_idx]] < scores[self.val[i]]:
            smallest = left_idx
        if r < self.size and scores[self.val[r]] < scores[self.val[smallest]]:
            smallest = r
        if smallest != i:
            self._swap(i, smallest)
            self.min_heapify(smallest, scores)
