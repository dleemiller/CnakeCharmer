"""Sum of all primes less than n using trial division.

Keywords: primes, sum, trial division, math, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def sum_primes(n: int) -> tuple:
    """Sum all primes less than n using trial division.

    Args:
        n: Upper bound (exclusive) for prime search.

    Returns:
        Tuple of (sum_of_primes, count_of_primes).
    """
    total = 0
    count = 0
    x = 2
    while x < n:
        is_prime = True
        if x == 2:
            is_prime = True
        elif x % 2 == 0:
            is_prime = False
        else:
            i = 3
            while i * i <= x:
                if x % i == 0:
                    is_prime = False
                    break
                i += 2
        if is_prime:
            total += x
            count += 1
        x += 1
    return (total, count)
