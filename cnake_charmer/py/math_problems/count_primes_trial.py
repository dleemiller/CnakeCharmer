"""Prime counting via trial division.

Keywords: math, primes, trial division, counting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x == 2:
        return True
    if x % 2 == 0:
        return False
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


@python_benchmark(args=(6000, True))
def count_primes_trial(limit: int, return_last: bool) -> tuple[int, int]:
    """Count primes up to limit and return (count, last_prime)."""
    count = 0
    last = 0
    for x in range(2, limit + 1):
        if _is_prime(x):
            count += 1
            last = x
    if not return_last:
        last = 0
    return count, last
