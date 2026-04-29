"""Linear congruential pseudo-random generator."""

from __future__ import annotations


def lin_con(a, b, m, length, initial_value=0):
    vals = [0] * length
    vals[0] = initial_value
    for x in range(1, length):
        vals[x] = (a * vals[x - 1] + b) % m
    return [v / float(m) for v in vals]
