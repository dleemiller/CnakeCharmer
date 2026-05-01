"""Array-backed binary decision tree helpers."""

from __future__ import annotations

import math


class ArrayTree:
    def __init__(self, max_depth=-1):
        self.max_depth = float("inf") if max_depth < 0 else int(max_depth)
        self.capacity = 0
        self.depth = 0
        self.n_nodes = 0
        self._data = []

    def plant(self):
        if self.max_depth != float("inf") and self.max_depth <= 10:
            init_capacity = 2 ** (self.max_depth + 1) - 1
        else:
            init_capacity = 2047
        self._resize(init_capacity)
        self.n_nodes = 1
        self._clean_node(0)
        return 0

    def sort(self, x):
        nid = 0
        while nid < self.n_nodes:
            node = self._data[nid]
            fid = node["fid"]
            if fid < 0:
                return nid
            go_left = x.get(fid, 0.0) <= node["theta"]
            nid = node["index"][0 if go_left else 1]
            if nid < 0:
                return -1
        return -1

    def split(self, nid, fid, theta):
        left = self.n_nodes
        right = self.n_nodes + 1
        if right >= self.capacity:
            self._resize(max(self.capacity * 2, right + 1))
        self._data[nid] = {"fid": int(fid), "theta": float(theta), "index": [left, right]}
        self._clean_node(left)
        self._clean_node(right)
        self.n_nodes += 2

    def _clean_node(self, nid):
        self._data[nid] = {"fid": -1, "theta": 0.0, "index": [-1, -1]}

    def _resize(self, size):
        if size <= self.capacity:
            return
        self._data.extend(
            {"fid": -1, "theta": 0.0, "index": [-1, -1]} for _ in range(size - self.capacity)
        )
        self.capacity = size
        if self.n_nodes:
            self.depth = int(math.floor(math.log2(self.n_nodes)))
