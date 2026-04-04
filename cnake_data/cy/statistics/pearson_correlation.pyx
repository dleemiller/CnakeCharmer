# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Pearson correlation coefficient between two deterministic sequences (Cython-optimized).

Keywords: statistics, pearson, correlation, numerical, cython, benchmark
"""

from libc.math cimport sin, sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def pearson_correlation(int n):
    """Compute Pearson correlation using typed loops and libc math.

    Args:
        n: Length of the sequences.

    Returns:
        Pearson correlation coefficient.
    """
    cdef int i
    cdef double sum_x = 0.0, sum_y = 0.0
    cdef double mean_x, mean_y
    cdef double cov = 0.0, var_x = 0.0, var_y = 0.0
    cdef double xi, yi, dx, dy
    cdef double angle

    # Pass 1: compute means
    for i in range(n):
        angle = i * 0.1
        sum_x += sin(angle) * 100.0
        sum_y += sin(angle + 0.5) * 100.0 + (i % 7)
    mean_x = sum_x / n
    mean_y = sum_y / n

    # Pass 2: covariance and standard deviations
    for i in range(n):
        angle = i * 0.1
        xi = sin(angle) * 100.0
        yi = sin(angle + 0.5) * 100.0 + (i % 7)
        dx = xi - mean_x
        dy = yi - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / sqrt(var_x * var_y)
