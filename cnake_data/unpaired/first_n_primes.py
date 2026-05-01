"""Compute first n primes via trial division."""

from __future__ import annotations


def primes(n_primes):
    if n_primes > 1000:
        raise ValueError("k <= 1000 required.")
    found = []
    n = 2
    while len(found) < n_primes:
        i = 0
        while i < len(found) and n % found[i] != 0:
            i += 1
        if i == len(found):
            found.append(n)
        n += 1
    return found
