"""Parallel prime counting up to n.

Keywords: math, primes, counting, parallel, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def prange_count_primes(n: int) -> int:
    """Count prime numbers in the range [2, n].

    Each number is tested independently via trial division.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Number of primes in [2, n].
    """
    count = 0
    for num in range(2, n + 1):
        is_prime = 1
        if num < 2:
            is_prime = 0
        elif num == 2:
            is_prime = 1
        elif num % 2 == 0:
            is_prime = 0
        else:
            d = 3
            while d * d <= num:
                if num % d == 0:
                    is_prime = 0
                    break
                d += 2
        count += is_prime

    return count
