"""
Sum of prime factors for all numbers from 2 to n.

Keywords: math, prime factorization, sieve, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def prime_factorization_sum(n: int) -> int:
    """Compute the total sum of prime factors (with multiplicity) for numbers 2..n.

    Uses a smallest-prime-factor sieve. For each number, repeatedly divide by
    its smallest prime factor to accumulate the factor sum.
    E.g. 12 = 2*2*3, factor sum = 2+2+3 = 7.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Total sum of all prime factors across all numbers 2..n.
    """
    # Build smallest prime factor sieve
    spf = list(range(n + 1))  # spf[i] = smallest prime factor of i
    i = 2
    while i * i <= n:
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
        i += 1

    # Sum prime factors for each number
    total = 0
    for num in range(2, n + 1):
        x = num
        while x > 1:
            total += spf[x]
            x //= spf[x]

    return total
