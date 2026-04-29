"""Spinlock-style circular insertion routines."""

from __future__ import annotations


def calculate_part1(n: int, turns: int) -> int:
    buf = [0]
    pos = 0
    for i in range(1, turns + 1):
        pos = (pos + n) % len(buf)
        buf.insert(pos + 1, i)
        pos += 1
    return buf[(pos + 1) % len(buf)]


def calculate_part2(n: int, turns: int) -> int:
    after_zero = 0
    current_pos = 0
    count = 1
    for i in range(1, turns + 1):
        current_pos = (current_pos + n) % count
        if current_pos == 0:
            after_zero = i
        count += 1
        current_pos += 1
    return after_zero
