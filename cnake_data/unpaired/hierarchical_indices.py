"""Generate hierarchical grid visitation indices."""

from __future__ import annotations


def hierarchical_indices(length):
    indices = []
    level = length // 2
    while level > 0:
        for i in range(level, length, 2 * level):
            for j in range(level, length, 2 * level):
                indices.append((i, j))
        for i in range(0, length, level):
            start = level if i % (2 * level) == 0 else 0
            for j in range(start, length, 2 * level):
                indices.append((i, j))
        level //= 2
    return indices
