# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Brute-force closest pair of points in 2D (Cython-optimized).

Keywords: geometry, closest pair, distance, brute force, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(4000,))
def closest_pair_distance(int n):
    """Find the minimum Euclidean distance between any two points using C arrays.

    Args:
        n: Number of points.

    Returns:
        The minimum distance between any two points.
    """
    cdef int i, j
    cdef double dx, dy, d, min_dist, xi, yi

    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))

    if xs == NULL or ys == NULL:
        if xs != NULL: free(xs)
        if ys != NULL: free(ys)
        raise MemoryError("Failed to allocate point arrays")

    for i in range(n):
        xs[i] = sin(i * 0.7) * 1000.0
        ys[i] = cos(i * 1.3) * 1000.0

    min_dist = 1e300
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            dx = xi - xs[j]
            dy = yi - ys[j]
            d = sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d

    free(xs)
    free(ys)
    return min_dist
