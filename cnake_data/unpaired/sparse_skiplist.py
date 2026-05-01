"""Sparse skip-list map with randomized heights."""

from __future__ import annotations

import random

MAX_HEIGHT = 32


class SkipNode:
    __slots__ = ("height", "index", "value", "next")

    def __init__(self, height, index, value):
        self.height = height
        self.index = index
        self.value = value
        self.next = [None] * height


class SparseSkipList:
    def __init__(self):
        self.size = 0
        self.height = 1
        self.head = SkipNode(MAX_HEIGHT, -1, -1.0)

    def _random_height(self):
        h = 1
        while random.getrandbits(1):
            h += 1
            if h >= MAX_HEIGHT:
                break
        return h

    def upsert(self, key, value):
        update = [self.head] * MAX_HEIGHT
        curr = self.head
        for level in range(self.height - 1, -1, -1):
            while curr.next[level] is not None and curr.next[level].index < key:
                curr = curr.next[level]
            update[level] = curr

        curr = curr.next[0]
        if curr is not None and curr.index == key:
            curr.value = value
            return

        new_h = self._random_height()
        if new_h > self.height:
            for level in range(self.height, new_h):
                update[level] = self.head
            self.height = new_h

        node = SkipNode(new_h, key, value)
        for level in range(new_h):
            node.next[level] = update[level].next[level]
            update[level].next[level] = node
        self.size += 1

    def find(self, key, default=None):
        curr = self.head
        for level in range(self.height - 1, -1, -1):
            while curr.next[level] is not None and curr.next[level].index < key:
                curr = curr.next[level]
        curr = curr.next[0]
        if curr is not None and curr.index == key:
            return curr.value
        return default
