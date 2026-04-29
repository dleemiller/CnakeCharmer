"""Open-address hash map with linear probing and resize checks."""

from __future__ import annotations


class HashMap:
    def __init__(self, init_pow2=8):
        self.max_items = 1 << init_pow2
        self.keys = [0] * self.max_items
        self.values = [0] * self.max_items
        self.used = [False] * self.max_items
        self.num_items = 0

    def _hash(self, key):
        return (key * 11400714819323198485) & (self.max_items - 1)

    def _grow_if_needed(self):
        if self.num_items / float(self.max_items) > 0.7:
            self._rehash(self.max_items * 2)

    def _rehash(self, new_cap):
        old = list(zip(self.keys, self.values, self.used, strict=False))
        self.max_items = new_cap
        self.keys = [0] * new_cap
        self.values = [0] * new_cap
        self.used = [False] * new_cap
        self.num_items = 0
        for k, v, u in old:
            if u:
                self.insert(k, v)

    def insert(self, key, value):
        self._grow_if_needed()
        h = self._hash(key)
        for i in range(self.max_items):
            idx = (h + i) & (self.max_items - 1)
            if self.used[idx]:
                if self.keys[idx] == key:
                    self.values[idx] = value
                    return
            else:
                self.keys[idx] = key
                self.values[idx] = value
                self.used[idx] = True
                self.num_items += 1
                return

    def get(self, key):
        h = self._hash(key)
        for i in range(self.max_items):
            idx = (h + i) & (self.max_items - 1)
            if not self.used[idx]:
                break
            if self.keys[idx] == key:
                return self.values[idx]
        raise KeyError("HashMap: key not found")
