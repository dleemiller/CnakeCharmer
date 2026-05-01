"""Recursive Fibonacci with per-index timing utility."""

from __future__ import annotations

import time


def fibo_rec(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibo_rec(n - 1) + fibo_rec(n - 2)


def nums(n, out_path="fiboRec_cy.txt"):
    with open(out_path, "w") as f:
        for i in range(n):
            ts = time.perf_counter()
            _num = fibo_rec(i)
            te = time.perf_counter()
            s = f"{i} {te - ts}\n"
            f.write(s)
