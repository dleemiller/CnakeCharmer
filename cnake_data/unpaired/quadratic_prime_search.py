"""Quadratic prime-run search utilities."""

from __future__ import annotations


def is_prime(val):
    if val < 2:
        return False
    i = 2
    while i * i <= val:
        if val % i == 0:
            return False
        i += 1
    return True


def quad(n, b, c):
    return n * n + b * n + c


def evaluator(bound=999):
    max_seq = 0
    best = (0, 0, 0)
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            k = 0
            while is_prime(quad(k, i, j)):
                k += 1
            if k > max_seq:
                max_seq = k
                best = (i, j, k)
    return best
