# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
2D prefix sums (summed area table) (Cython-optimized).

Keywords: numerical, prefix sum, 2D, summed area table, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def prefix_sum_2d(int n):
    """Compute 2D prefix sums for an n x n grid using flat C array."""
    cdef long long *P = <long long *>malloc(n * n * sizeof(long long))
    cdef int i, j
    cdef long long g, above, left, diag

    if P == NULL:
        raise MemoryError("Failed to allocate array")

    with nogil:
        for i in range(n):
            for j in range(n):
                g = ((i * 1009 + j * 2003 + 42) * 17 + 137) % 256
                above = P[(i - 1) * n + j] if i > 0 else 0
                left = P[i * n + (j - 1)] if j > 0 else 0
                diag = P[(i - 1) * n + (j - 1)] if (i > 0 and j > 0) else 0
                P[i * n + j] = g + above + left - diag

    result = (
        P[(n - 1) * n + (n - 1)],
        P[(n // 2) * n + (n // 3)],
        P[(n // 3) * n + (n // 2)],
        P[(n - 1) * n + 0] + P[0 * n + (n - 1)],
    )
    free(P)
    return result
