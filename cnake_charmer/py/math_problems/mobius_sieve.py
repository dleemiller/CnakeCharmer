"""Compute the Mobius function for 1..n using a sieve and return the sum.

Keywords: mobius, number theory, sieve, multiplicative function, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def mobius_sieve(n: int) -> int:
    """Compute Mobius function mu(k) for k=1..n and return sum of all values.

    mu(1)=1, mu(n)=0 if n has a squared prime factor,
    mu(n)=(-1)^k if n is product of k distinct primes.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Sum of mu(1) + mu(2) + ... + mu(n).
    """
    mu = [0] * (n + 1)
    mu[1] = 1

    # Sieve: for each prime p, multiply mu by -1 for multiples of p,
    # and set mu to 0 for multiples of p^2.
    is_prime = [True] * (n + 1)

    for i in range(2, n + 1):
        if is_prime[i]:
            # i is prime: mu[i] = -1
            mu[i] = -1
            # Mark composites
            for j in range(2 * i, n + 1, i):
                is_prime[j] = False
            # For all multiples of i, multiply mu by -1
            for j in range(2 * i, n + 1, i):
                mu[j] = -mu[j]
            # For multiples of i^2, set mu to 0
            i2 = i * i
            for j in range(i2, n + 1, i2):
                mu[j] = 0

    total = 0
    for k in range(1, n + 1):
        total += mu[k]
    return total
