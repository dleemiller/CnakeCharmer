"""
Sieve of Eratosthenes to find all primes up to n.

Keywords: sieve, eratosthenes, primes, math, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def sieve_of_eratosthenes(n: int) -> list[int]:
    """Find all prime numbers up to n using the Sieve of Eratosthenes.

    Args:
        n: Upper bound (inclusive) for prime search.

    Returns:
        List of all primes up to n.
    """
    if n < 2:
        return []

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

    result = []
    for i in range(2, n + 1):
        if is_prime[i]:
            result.append(i)

    return result
