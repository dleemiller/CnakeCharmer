"""Sieve of Eratosthenes prime generator."""

from __future__ import annotations

import math


def sieve(n):
    arr = [0] * (n + 1)
    for i in range(2, n + 1):
        arr[i] = 1

    sqrt_n = int(math.sqrt(n))
    for i in range(2, sqrt_n + 1):
        if arr[i] == 1:
            j = i * i
            while j <= n:
                arr[j] = 0
                j += i

    return [i for i in range(2, n + 1) if arr[i] == 1]
