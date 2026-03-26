"""
Compute Euler's totient function for all numbers 1..n using a sieve.

Keywords: math, euler, totient, sieve, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def euler_totient_sieve(n: int) -> list:
    """Compute Euler's totient (phi) for all numbers from 1 to n using a sieve.

    Uses the standard sieve approach: initialize phi[i] = i, then for each
    prime p, update all multiples of p.

    Args:
        n: Upper bound (inclusive) for the sieve.

    Returns:
        List of ints of length n, where result[i] = phi(i + 1).
    """
    if n < 1:
        return []

    phi = list(range(n + 1))

    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i

    return phi[1:]
