# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute all pairwise Euclidean distances between n 2D points (Cython-optimized).

Keywords: numerical, euclidean distance, pairwise, geometry, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def euclidean_distances(int n):
    """Compute the sum of all pairwise Euclidean distances using C arrays and C sqrt."""
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    if not xs or not ys:
        if xs: free(xs)
        if ys: free(ys)
        raise MemoryError()

    cdef int i, j
    cdef double total = 0.0
    cdef double dx, dy

    for i in range(n):
        xs[i] = i * 0.7
        ys[i] = i * 1.3

    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            total += sqrt(dx * dx + dy * dy)

    free(xs)
    free(ys)
    return total
