"""Sieve of Eratosthenes collecting primes into a list.

Keywords: sieve, primes, Eratosthenes, list, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(500000,))
def stl_vector_sieve(n: int) -> tuple:
    """Sieve of Eratosthenes up to n, collect primes, return discriminating tuple.

    Args:
        n: Upper bound (inclusive) for the sieve.

    Returns:
        Tuple of (num_primes, weighted_sum_mod) where weighted_sum_mod is
        sum(prime[i] * (i+1) for i in range(len(primes))) % (10**9 + 7).
    """
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = 0
    if n >= 1:
        sieve[1] = 0

    i = 2
    while i * i <= n:
        if sieve[i]:
            j = i * i
            while j <= n:
                sieve[j] = 0
                j += i
        i += 1

    primes = []
    for k in range(2, n + 1):
        if sieve[k]:
            primes.append(k)

    num_primes = len(primes)
    weighted_sum = 0
    for idx in range(num_primes):
        weighted_sum = (weighted_sum + primes[idx] * (idx + 1)) % MOD

    return (num_primes, weighted_sum)
