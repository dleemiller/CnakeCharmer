from __future__ import annotations


def sumint_1_to_n(i: int) -> int:
    s = 0
    for x in range(i):
        s += x
    return s
