"""Sum all primes below n using trial division.

Keywords: algorithms, primes, trial division, sum, count
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def prime_sum(n: int) -> tuple[int, int]:
    """Sum and count all prime numbers less than n using trial division.

    Args:
        n: Upper bound (exclusive).

    Returns:
        (prime_sum, prime_count) — sum and count of primes < n.
    """

    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        limit = int(math.sqrt(x)) + 1
        return all(x % d != 0 for d in range(3, limit, 2))

    total = 0
    count = 0
    for x in range(2, n):
        if is_prime(x):
            total += x
            count += 1

    return (total, count)
