"""Polynomial evaluation with uniform workload (static schedule).

Keywords: numerical, polynomial, prange, static, parallel, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def prange_schedule_static(n: int) -> float:
    """Evaluate a degree-8 polynomial at n points, return sum.

    Each point x_i = (i * 13 + 7) % 10000 / 10000.0.
    Polynomial: 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 6x^5 + 7x^6 + 8x^7.
    All elements take the same time (uniform workload).

    Args:
        n: Number of evaluation points.

    Returns:
        Sum of all polynomial evaluations.
    """
    total = 0.0
    for i in range(n):
        x = ((i * 13 + 7) % 10000) / 10000.0
        # Horner's method for degree-7 polynomial
        val = 8.0
        val = val * x + 7.0
        val = val * x + 6.0
        val = val * x + 5.0
        val = val * x + 4.0
        val = val * x + 3.0
        val = val * x + 2.0
        val = val * x + 1.0
        total += val
    return total
