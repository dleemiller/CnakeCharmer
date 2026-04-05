"""Combine prime generation and bubble-sort swap counting.

Adapted from The Stack v2 Cython candidate:
- blob_id: 81fffe9e4fe42d5fb755660a3ba982417a67b613
- filename: refactor_mod.pyx

Keywords: algorithms, primes, bubble sort, swap count, checksum
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(7000, 1024))
def stack_prime_bubble_combo(limit: int, sort_n: int) -> tuple:
    """Generate primes by trial division, then bubble-sort transformed values."""
    primes = []
    for n in range(2, limit + 1):
        ok = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                ok = False
                break
            d += 1
        if ok:
            primes.append(n)

    m = min(sort_n, len(primes))
    vals = [((primes[i] * 37 + 11) % 10007) for i in range(m)]

    swaps = 0
    for i in range(m):
        for j in range(0, m - i - 1):
            if vals[j] > vals[j + 1]:
                vals[j], vals[j + 1] = vals[j + 1], vals[j]
                swaps += 1

    checksum = 0
    for i, v in enumerate(vals):
        checksum = (checksum + v * (i + 3)) & 0xFFFFFFFF

    last_prime = primes[-1] if primes else 0
    return (len(primes), last_prime, swaps, checksum)
