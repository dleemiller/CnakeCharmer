"""Count primes in the first n integers using trial division.

Keywords: prime, counting, math, cpdef, standalone function, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def is_prime(n: int) -> bool:
    """Check if n is prime using trial division."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


@python_benchmark(args=(50000,))
def cpdef_is_prime_count(n: int) -> int:
    """Count how many integers in [0, n) are prime.

    Args:
        n: Upper bound (exclusive) for counting primes.

    Returns:
        Number of primes in [0, n).
    """
    count = 0
    for i in range(n):
        if is_prime(i):
            count += 1
    return count
