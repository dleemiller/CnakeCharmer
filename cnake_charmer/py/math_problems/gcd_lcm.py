"""
Compute GCD and LCM sums over all pairs in range(1, n).

Keywords: gcd, lcm, math, pairs, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def gcd_lcm(n: int) -> tuple:
    """Compute GCD and LCM of all pairs (i, j) where 1 <= i < j < n, and sum them.

    Uses the Euclidean algorithm for GCD and derives LCM as i*j // gcd(i,j).

    Args:
        n: Upper bound (exclusive) for pair generation.

    Returns:
        Tuple of (gcd_sum, lcm_sum).
    """
    gcd_sum = 0
    lcm_sum = 0

    for i in range(1, n):
        for j in range(i + 1, n):
            a = i
            b = j
            while b != 0:
                a, b = b, a % b
            g = a
            gcd_sum += g
            lcm_sum += i * j // g

    return (gcd_sum, lcm_sum)
