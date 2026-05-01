from __future__ import annotations


def is_prime_trial(n: int) -> bool:
    if n < 3:
        return n == 2
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def primes_list(n: int) -> list[int]:
    if n <= 2:
        return []
    sieve = [True] * n
    i = 3
    while i * i < n:
        if sieve[i]:
            step = 2 * i
            start = i * i
            sieve[start:n:step] = [False] * (((n - start - 1) // step) + 1)
        i += 2
    return [2] + [i for i in range(3, n, 2) if sieve[i]]
