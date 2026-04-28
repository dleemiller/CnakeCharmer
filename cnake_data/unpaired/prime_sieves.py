"""Prime generation helpers (sieve and nth-prime)."""

from __future__ import annotations

from math import sqrt

import numpy as np


def primesfrom2to(n: int):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(sqrt(n)) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def primesfrom3to(n: int):
    sieve = np.ones(n // 2, dtype=bool)
    for i in range(3, int(sqrt(n)) + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2 :: i] = False
    return 2 * np.nonzero(sieve)[0][1::] + 1


def nth_prime(n: int):
    def _is_prime(num: int):
        return all(num % i != 0 for i in range(3, int(sqrt(num)) + 1, 2))

    if n <= 0:
        raise ValueError("n must be > 0")
    if n == 1:
        return 2
    if n == 2:
        return 3

    prime_count = 2
    num = 3
    while prime_count != n:
        num += 2
        if _is_prime(num):
            prime_count += 1
    return num
