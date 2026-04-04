"""
Evaluate renormalized associated Legendre polynomials at many x values and sum.

Keywords: legendre, polynomial, math, recurrence, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def assoc_legendre_sum(n: int) -> tuple:
    """Evaluate renormalized associated Legendre P_10^3(x) at n points and sum.

    Uses the recurrence relation to compute the renormalized associated
    Legendre polynomial for l=10, m=3 at n evenly-spaced x values in
    [-0.99, 0.99], then returns (total_sum, value_at_midpoint).

    Args:
        n: Number of evaluation points.

    Returns:
        Tuple of (sum of all values, value at x=0).
    """
    pi = 3.14159265358979323846
    ell = 10
    m = 3

    def _sqrt(val):
        if val <= 0.0:
            return 0.0
        x = val
        for _ in range(20):
            x = (x + val / x) * 0.5
        return x

    def _assoc_legendre(x):
        pmm = 1.0
        omx2 = (1.0 - x) * (1.0 + x)
        fact = 1.0
        for _i in range(1, m + 1):
            pmm *= omx2 * fact / (fact + 1.0)
            fact += 2.0
        pmm = _sqrt((2 * m + 1) * pmm / (4.0 * pi))
        if m & 1:
            pmm = -pmm
        # ell != m, so continue
        pmmp1 = x * _sqrt(2.0 * m + 3.0) * pmm
        # ell != m+1, so continue
        oldfact = _sqrt(2.0 * m + 3.0)
        pll = 0.0
        for ll in range(m + 2, ell + 1):
            fact = _sqrt((4.0 * ll * ll - 1.0) / (ll * ll - m * m))
            pll = (x * pmmp1 - pmm / oldfact) * fact
            oldfact = fact
            pmm = pmmp1
            pmmp1 = pll
        return pll

    total = 0.0
    mid_val = 0.0
    mid_idx = n // 2

    for i in range(n):
        x = -0.99 + 1.98 * i / (n - 1) if n > 1 else 0.0
        val = _assoc_legendre(x)
        total += val
        if i == mid_idx:
            mid_val = val

    return (total, mid_val)
