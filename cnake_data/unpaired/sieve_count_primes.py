from __future__ import annotations


def sieve_flags(n: int) -> list[bool]:
    if n < 2:
        return [False] * max(0, n + 1)
    p = [True] * (n + 1)
    p[0] = p[1] = False
    k = 2
    while k * k <= n:
        if p[k]:
            m = k * k
            while m <= n:
                p[m] = False
                m += k
        k += 1
    return p


def count_primes(n: int) -> int:
    return sum(1 for x in sieve_flags(n) if x)
