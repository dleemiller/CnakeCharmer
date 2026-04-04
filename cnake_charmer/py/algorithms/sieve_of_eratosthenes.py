"""Sieve of Eratosthenes to find all primes up to n.

Keywords: algorithms, sieve, primes, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def sieve_of_eratosthenes(n: int) -> tuple:
    """Find all primes up to n using the Sieve of Eratosthenes.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Tuple of (prime_count, prime_sum % 10**9).
    """
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    if n >= 1:
        is_prime[1] = 0

    i = 2
    while i * i <= n:
        if is_prime[i]:
            j = i * i
            while j <= n:
                is_prime[j] = 0
                j += i
        i += 1

    prime_count = 0
    prime_sum = 0
    for k in range(2, n + 1):
        if is_prime[k]:
            prime_count += 1
            prime_sum += k

    return (prime_count, prime_sum % (10**9))
