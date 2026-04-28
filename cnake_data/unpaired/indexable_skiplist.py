import math
import random


class Node:
    def __init__(self, value, nxt, width):
        self.value = value
        self.next = nxt
        self.width = width


NIL = Node(float("inf"), [], [])


class IndexableSkiplist:
    """Sorted collection with O(log n)-style updates and rank lookup."""

    def __init__(self, expected_size=100):
        self.size = 0
        self.maxlevels = int(1 + math.log(max(expected_size, 2), 2))
        self.head = Node(float("nan"), [NIL] * self.maxlevels, [1] * self.maxlevels)

    def __len__(self):
        return self.size

    def get(self, i):
        node = self.head
        i += 1
        for level in range(self.maxlevels - 1, -1, -1):
            while node.width[level] <= i:
                i -= node.width[level]
                node = node.next[level]
        return node.value

    def insert(self, value):
        chain = [None] * self.maxlevels
        steps_at_level = [0] * self.maxlevels
        node = self.head

        for level in range(self.maxlevels - 1, -1, -1):
            nxt = node.next[level]
            while nxt.value <= value:
                steps_at_level[level] += node.width[level]
                node = nxt
                nxt = node.next[level]
            chain[level] = node

        # random level count
        r = max(random.random(), 1e-12)
        d = min(self.maxlevels, 1 - int(math.log(r, 2)))
        newnode = Node(value, [None] * d, [None] * d)
        steps = 0

        for level in range(d):
            prev = chain[level]
            newnode.next[level] = prev.next[level]
            prev.next[level] = newnode
            newnode.width[level] = prev.width[level] - steps
            prev.width[level] = steps + 1
            steps += steps_at_level[level]

        for level in range(d, self.maxlevels):
            chain[level].width[level] += 1

        self.size += 1

    def remove(self, value):
        chain = [None] * self.maxlevels
        node = self.head
        for level in range(self.maxlevels - 1, -1, -1):
            nxt = node.next[level]
            while nxt.value < value:
                node = nxt
                nxt = node.next[level]
            chain[level] = node

        target = chain[0].next[0]
        if target.value != value:
            raise KeyError("Not Found")

        d = len(target.next)
        for level in range(d):
            prev = chain[level]
            prev.width[level] += target.width[level] - 1
            prev.next[level] = target.next[level]

        for level in range(d, self.maxlevels):
            chain[level].width[level] -= 1

        self.size -= 1
