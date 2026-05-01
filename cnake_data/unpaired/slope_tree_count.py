from __future__ import annotations


def count_trees(lines: list[str], dx: int = 3, dy: int = 1) -> int:
    if not lines:
        return 0
    w = len(lines[0])
    h = len(lines)
    x = 0
    c = 0
    for y in range(0, h, dy):
        if lines[y][x % w] == "#":
            c += 1
        x += dx
    return c
