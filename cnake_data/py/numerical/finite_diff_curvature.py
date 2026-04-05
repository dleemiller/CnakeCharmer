"""Summarize second-difference curvature over a deterministic series.

Keywords: numerical, finite differences, curvature, discrete derivative, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def finite_diff_curvature(n: int) -> tuple:
    """Compute second differences and return magnitude statistics."""
    if n < 3:
        return (0, 0, 0)

    y0 = ((0 * 0 * 17 + 0 * 31 + 7) % 1000) - 500
    y1 = ((1 * 1 * 17 + 1 * 31 + 7) % 1000) - 500

    sum_abs = 0
    max_abs = 0
    signed_sum = 0

    for i in range(2, n):
        yi = ((i * i * 17 + i * 31 + 7) % 1000) - 500
        d2 = yi - 2 * y1 + y0
        ad2 = d2 if d2 >= 0 else -d2
        sum_abs += ad2
        if ad2 > max_abs:
            max_abs = ad2
        signed_sum += d2 * ((i % 7) - 3)
        y0, y1 = y1, yi

    return (sum_abs, max_abs, signed_sum)
