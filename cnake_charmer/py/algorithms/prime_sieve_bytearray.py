"""Sieve of Eratosthenes using a bytearray for compact prime flagging.

Marks composite numbers in a half-sieve (odd numbers only) using a bytearray,
then counts the surviving primes. This approach avoids storing all primes
in a list and uses minimal memory per candidate.

Keywords: algorithms, prime sieve, Eratosthenes, bytearray, number theory, benchmark
"""

from math import sqrt

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def prime_sieve_bytearray(limit: int) -> tuple:
    """Count primes up to limit using a bytearray sieve.

    Uses the sieve of Eratosthenes on odd numbers only, storing flags in a
    bytearray. After sieving, counts the primes and computes the sum of all
    primes for validation.

    Args:
        limit: Upper bound for the sieve (inclusive).

    Returns:
        Tuple of (prime_count, prime_sum).
    """
    if limit < 2:
        return (0, 0)

    # Only store odd numbers: index i represents number 2*i + 1
    size = (limit + 1) // 2
    bits = bytearray(b"\x01") * size

    factor = 1
    q = sqrt(limit) / 2.0

    while factor <= q:
        # Find next prime (next set bit)
        for index in range(factor, size):
            if bits[index]:
                factor = index
                break

        # Mark multiples starting at factor^2
        start = 2 * factor * (factor + 1)
        step = factor * 2 + 1

        idx = start
        while idx < size:
            bits[idx] = 0
            idx += step

        factor += 1

    # Count primes and compute sum
    prime_count = 1  # count 2
    prime_sum = 2
    for i in range(1, size):
        if bits[i]:
            prime_count += 1
            prime_sum += 2 * i + 1

    return (prime_count, prime_sum)
