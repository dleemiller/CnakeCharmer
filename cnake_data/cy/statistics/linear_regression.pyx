# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simple linear regression: compute slope from deterministic points (Cython-optimized).

Keywords: statistics, linear regression, slope, intercept, least squares, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def linear_regression(int n):
    """Compute slope of simple linear regression on n deterministic points.

    Args:
        n: Number of data points.

    Returns:
        The slope of the regression line.
    """
    cdef int i
    cdef double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0
    cdef double x, y, denom

    for i in range(n):
        x = <double>i
        y = ((i * 17 + 5) % 1000) / 10.0
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom
