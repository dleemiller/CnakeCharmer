"""Iterative and recursive Fibonacci implementations."""

from __future__ import annotations


def fib_iter(n):
    a, b = 0.0, 1.0
    for _ in range(n):
        a, b = a + b, a
    return a


def fib_rec(n):
    if n == 0 or n == 1:
        return float(n)
    return fib_rec(n - 1) + fib_rec(n - 2)
