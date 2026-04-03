"""Sieve of Eratosthenes for prime counting.

Keywords: grpo, math, primes, sieve, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def sieve_primes(n: int) -> tuple:
    """Count primes up to n using the Sieve of Eratosthenes.

    Returns (prime_count, sum of first 100 primes, largest prime <= n).

    Args:
        n: Upper bound for prime search.

    Returns:
        Tuple of (count, sum_first_100, largest_prime).
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n > 0:
        is_prime[1] = False

    i = 2
    while i * i <= n:
        if is_prime[i]:
            j = i * i
            while j <= n:
                is_prime[j] = False
                j += i
        i += 1

    count = 0
    sum_first_100 = 0
    largest = 0
    found = 0

    for i in range(2, n + 1):
        if is_prime[i]:
            count += 1
            largest = i
            if found < 100:
                sum_first_100 += i
                found += 1

    return (count, sum_first_100, largest)
