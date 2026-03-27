"""Sieve of Eratosthenes returning count, largest prime, and sum mod 10^9+7.

Keywords: sieve, primes, number theory, math, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def prime_sieve(n: int) -> tuple:
    """Sieve primes up to n and return summary statistics.

    Args:
        n: Upper bound (inclusive) for prime search.

    Returns:
        Tuple of (count_primes, largest_prime, sum_of_primes_mod_1000000007).
    """
    if n < 2:
        return (0, 0, 0)

    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False

    i = 2
    while i * i <= n:
        if is_prime[i]:
            j = i * i
            while j <= n:
                is_prime[j] = False
                j += i
        i += 1

    MOD = 1000000007
    count = 0
    largest = 0
    total = 0
    for i in range(2, n + 1):
        if is_prime[i]:
            count += 1
            largest = i
            total = (total + i) % MOD

    return (count, largest, total)
