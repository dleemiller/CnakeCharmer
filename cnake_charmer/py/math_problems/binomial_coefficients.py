"""Compute row n of binomial coefficients mod a prime using Pascal's triangle.

Keywords: binomial, pascal, triangle, combinatorics, coefficients, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(3000,))
def binomial_coefficients(n: int) -> tuple:
    """Compute row n of Pascal's triangle mod 10^9+7 and return summary statistics.

    Builds Pascal's triangle row by row up to row n, using the identity
    C(n, k) = C(n-1, k-1) + C(n-1, k), all reduced mod 10^9+7.

    Args:
        n: The row number to compute (0-indexed).

    Returns:
        Tuple of (sum of row mod p, middle value C(n, n//2) mod p,
        weighted checksum) where weighted checksum is
        sum of (k * C(n, k)) mod p for all k.
    """
    mod = MOD

    # Build row n iteratively, all mod p
    prev = [0] * (n + 1)
    prev[0] = 1

    for i in range(1, n + 1):
        curr = [0] * (n + 1)
        curr[0] = 1
        for k in range(1, i + 1):
            curr[k] = (prev[k - 1] + prev[k]) % mod
        prev = curr

    row_sum = 0
    for k in range(n + 1):
        row_sum = (row_sum + prev[k]) % mod

    middle = prev[n // 2]

    checksum = 0
    for k in range(n + 1):
        checksum = (checksum + k * prev[k]) % mod

    return (row_sum, middle, checksum)
