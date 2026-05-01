from __future__ import annotations


def increment_n(x: int) -> int:
    y = 0
    for _ in range(x):
        y += 1
    return y
