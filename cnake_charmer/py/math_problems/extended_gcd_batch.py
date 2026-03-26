"""Compute extended GCD for n deterministic pairs and sum |x| values.

Keywords: extended euclidean, gcd, number theory, bezout, batch, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _extended_gcd(a, b):
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b). Iterative."""
    old_r, r = a, b
    old_s, s = 1, 0
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    # old_r = gcd, old_s = x, y = (old_r - a*old_s) / b
    return old_r, old_s


@python_benchmark(args=(1000000,))
def extended_gcd_batch(n: int) -> int:
    """Compute extended GCD for n pairs and sum absolute x coefficients.

    For each i in 0..n-1, compute extended GCD of a=(i*7+3)%1000+1 and
    b=(i*13+7)%1000+1, yielding ax+by=gcd(a,b). Return sum of |x|.

    Args:
        n: Number of pairs to process.

    Returns:
        Sum of |x| for all pairs.
    """
    total = 0
    for i in range(n):
        a = (i * 7 + 3) % 1000 + 1
        b = (i * 13 + 7) % 1000 + 1
        _, x = _extended_gcd(a, b)
        total += abs(x)
    return total
