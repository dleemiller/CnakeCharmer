"""
Compute sum of Euler's totient function phi(1) + phi(2) + ... + phi(n).

Keywords: math, euler, totient, sieve, sum, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def totient_sum(n: int) -> int:
    """Compute sum of phi(1) + phi(2) + ... + phi(n) using a sieve.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Sum of all totient values from 1 to n.
    """
    if n < 1:
        return 0

    phi = list(range(n + 1))

    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i

    total = 0
    for i in range(1, n + 1):
        total += phi[i]

    return total
