"""Secant method root finding.

Keywords: numerical, root finding, secant method, iterative, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def secant_method(n: int) -> float:
    """Find roots of f(x) = x^3 - 2x - 5 using the secant method.

    For n different starting points x0 = i * 0.1, uses the secant method
    with x1 = x0 + 1.0 to find a root. Returns the sum of all roots found.

    Args:
        n: Number of starting points to try.

    Returns:
        Sum of all roots found.
    """
    total = 0.0

    for i in range(n):
        x0 = i * 0.1
        x1 = x0 + 1.0

        f0 = x0 * x0 * x0 - 2.0 * x0 - 5.0
        f1 = x1 * x1 * x1 - 2.0 * x1 - 5.0

        for _ in range(50):
            if abs(f1 - f0) < 1e-15:
                break
            x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0 = x1
            f0 = f1
            x1 = x_new
            f1 = x1 * x1 * x1 - 2.0 * x1 - 5.0
            if abs(f1) < 1e-12:
                break

        total += x1

    return total
