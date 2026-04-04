"""Variable-iteration convergence with dynamic schedule.

Keywords: numerical, Newton's method, prange, dynamic, parallel, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def prange_schedule_dynamic(n: int) -> float:
    """Compute cube roots via Newton iterations, return sum.

    For each i in 1..n, find cbrt(i) using Newton's method
    for f(x) = x^3 - i, iterating until |x^3 - i| < 1e-12
    or max 50 iterations. Some values converge fast, others
    slow, making dynamic scheduling beneficial.

    Args:
        n: Number of values to compute cube roots for.

    Returns:
        Sum of all computed cube roots.
    """
    total = 0.0
    for i in range(1, n + 1):
        val = float(i)
        x = val / 3.0  # initial guess
        for _ in range(50):
            x3 = x * x * x
            if abs(x3 - val) < 1e-12:
                break
            x = x - (x3 - val) / (3.0 * x * x)
        total += x
    return total
