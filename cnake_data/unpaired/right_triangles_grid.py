"""Count right triangles with one vertex at origin on grid."""

from __future__ import annotations


def is_right_triangle(p1, p2):
    p0 = (0, 0)
    if p0 == p1 or p0 == p2 or p1 == p2:
        return False

    p0p1 = p1[0] * p1[0] + p1[1] * p1[1]
    p0p2 = p2[0] * p2[0] + p2[1] * p2[1]
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    p1p2 = dx * dx + dy * dy

    a, b, c = sorted([p0p1, p0p2, p1p2])
    return a + b == c


def count_right_triangles(limit=50):
    duplicates = set()
    for x in range(limit + 1):
        for y in range(limit + 1):
            p1 = (x, y)
            for i in range(limit + 1):
                for j in range(limit + 1):
                    p2 = (i, j)
                    if (p1, p2) not in duplicates and is_right_triangle(p1, p2):
                        duplicates.add((p2, p1))
    return len(duplicates)
