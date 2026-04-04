"""Compute square roots via Newton's method with GIL release.

Keywords: numerical, Newton's method, square root, nogil, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def nogil_newton_roots(n: int) -> float:
    """Compute sqrt of n values via Newton's method, return sum.

    For each i in 1..n, compute sqrt(i) using 10 Newton iterations
    starting from i/2. Returns the sum of all computed roots.

    Args:
        n: Number of values to compute square roots for.

    Returns:
        Sum of all computed square roots.
    """
    total = 0.0
    for i in range(1, n + 1):
        x = i * 0.5
        val = float(i)
        for _ in range(10):
            x = 0.5 * (x + val / x)
        total += x
    return total
