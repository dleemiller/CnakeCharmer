"""
Sum primes via trial division.

Sourced from SFT DuckDB blob: 14834b7d32cd6d788819f5842484cf3bd5313aa9
Keywords: prime, trial division, prime summation, algorithms
"""

from cnake_charmer.benchmarks import python_benchmark


def _is_prime_trial(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@python_benchmark(args=(150000, 2, 1))
def prime_sum_trial_division(limit: int, start: int, step: int) -> tuple:
    """Return (sum_primes, count_primes, last_prime) over arithmetic scan."""
    total = 0
    count = 0
    last_prime = -1

    for x in range(start, limit, step):
        if _is_prime_trial(x):
            total += x
            count += 1
            last_prime = x

    return (total, count, last_prime)
