"""Batch bisection root finding.

Keywords: bisection, root finding, batch, trigonometric, optimization, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def bisection_batch(n: int) -> float:
    """Find roots of f(x) = sin(x) - x/k for n values of k.

    k = i + 1 for i in range(n). Bisection on [0, pi], 50 iterations each.
    Returns sum of all roots found.

    Args:
        n: Number of different k values to solve.

    Returns:
        Sum of all roots found.
    """
    pi = math.pi
    total = 0.0

    for i in range(n):
        k = float(i + 1)
        lo = 0.0
        hi = pi

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            fmid = math.sin(mid) - mid / k
            if fmid > 0.0:
                lo = mid
            else:
                hi = mid

        total += 0.5 * (lo + hi)

    return total
