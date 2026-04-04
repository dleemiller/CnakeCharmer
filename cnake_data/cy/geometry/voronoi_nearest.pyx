# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of nearest-neighbor distances (Cython-optimized).

Keywords: geometry, voronoi, nearest neighbor, distance, brute force, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def voronoi_nearest(int n):
    """Compute sum of nearest-neighbor distances for n 2D points.

    Uses C arrays and libc math for O(n^2) brute-force search.

    Args:
        n: Number of points.

    Returns:
        Sum of nearest-neighbor distances.
    """
    cdef int i, j
    cdef double dx, dy, d, min_dist, xi, yi, total

    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    if xs == NULL or ys == NULL:
        if xs != NULL: free(xs)
        if ys != NULL: free(ys)
        raise MemoryError()

    for i in range(n):
        xs[i] = sin(i * 0.7) * 100.0
        ys[i] = cos(i * 1.3) * 100.0

    total = 0.0
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        min_dist = 1e300
        for j in range(n):
            if i == j:
                continue
            dx = xi - xs[j]
            dy = yi - ys[j]
            d = sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
        total = total + min_dist

    free(xs)
    free(ys)
    return total
