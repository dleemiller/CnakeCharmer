from __future__ import annotations


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def count_primes_upto(n: int) -> int:
    c = 0
    for k in range(2, n + 1):
        if is_prime(k):
            c += 1
    return c
