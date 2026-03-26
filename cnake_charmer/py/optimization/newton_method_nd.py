"""Newton's method root finding for multiple starting points.

Keywords: newton, root finding, cubic, optimization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def newton_method_nd(n: int) -> int:
    """Find roots of f(x) = x^3 - 2x - 5 using Newton's method.

    Tests n different starting points x0 = i*0.01 - 2.5.
    Counts how many converge (|f(x)| < 1e-10 within 50 iterations).

    Args:
        n: Number of starting points to test.

    Returns:
        Count of converged roots.
    """
    count = 0

    for i in range(n):
        x = i * 0.01 - 2.5
        converged = 0

        for _ in range(50):
            fx = x * x * x - 2.0 * x - 5.0
            fpx = 3.0 * x * x - 2.0
            if abs(fpx) < 1e-30:
                break
            x = x - fx / fpx
            if abs(fx) < 1e-10:
                converged = 1
                break

        count += converged

    return count
