# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute pooled within-row variance of a 2D matrix (Cython-optimized).

For each row, subtract row mean, square, sum all, divide by total elements.

Keywords: statistics, variance, gene, row, matrix, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(8000,))
def gene_row_variance(int n):
    """Compute pooled within-row variance of an n x 20 matrix."""
    cdef int cols = 20
    cdef int i, j
    cdef double row_sum, val, diff
    cdef double row_means_sum = 0.0
    cdef double total_sq = 0.0
    cdef double variance
    cdef double *row_means = <double *>malloc(n * sizeof(double))
    if not row_means:
        raise MemoryError()

    with nogil:
        for i in range(n):
            row_sum = 0.0
            for j in range(cols):
                row_sum += ((i * 7 + j * 13 + 42) % 1000) / 100.0
            row_means[i] = row_sum / cols
            row_means_sum += row_means[i]

        for i in range(n):
            for j in range(cols):
                val = ((i * 7 + j * 13 + 42) % 1000) / 100.0
                diff = val - row_means[i]
                total_sq += diff * diff

        variance = total_sq / (n * cols)
        free(row_means)

    return (variance, row_means_sum)
