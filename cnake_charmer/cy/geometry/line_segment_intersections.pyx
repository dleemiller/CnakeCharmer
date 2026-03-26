# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count intersections among n line segments using brute force (Cython-optimized).

Keywords: geometry, line segment, intersection, cross product, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def line_segment_intersections(int n):
    """Count intersecting segment pairs using C arrays and typed cross products.

    Args:
        n: Number of line segments.

    Returns:
        Number of intersecting segment pairs.
    """
    cdef int i, j_idx, k
    cdef double ex, ey, fx, fy, gx, gy, hx, hy
    cdef double d1, d2, d3, d4
    cdef double dx_i, dy_i, dx_j, dy_j
    cdef double ax_i, ay_i
    cdef int count = 0

    cdef double *ax = <double *>malloc(n * sizeof(double))
    cdef double *ay = <double *>malloc(n * sizeof(double))
    cdef double *bx = <double *>malloc(n * sizeof(double))
    cdef double *by = <double *>malloc(n * sizeof(double))

    if ax == NULL or ay == NULL or bx == NULL or by == NULL:
        if ax != NULL: free(ax)
        if ay != NULL: free(ay)
        if bx != NULL: free(bx)
        if by != NULL: free(by)
        raise MemoryError("Failed to allocate segment arrays")

    for i in range(n):
        ax[i] = i * 0.7
        ay[i] = i * 1.3
        k = (i * 3 + 1) % n
        bx[i] = k * 0.7
        by[i] = k * 1.3

    for i in range(n):
        ax_i = ax[i]
        ay_i = ay[i]
        dx_i = bx[i] - ax_i
        dy_i = by[i] - ay_i
        for j_idx in range(i + 1, n):
            ex = ax[j_idx] - ax_i
            ey = ay[j_idx] - ay_i
            fx = bx[j_idx] - ax_i
            fy = by[j_idx] - ay_i

            d1 = dx_i * ey - dy_i * ex
            d2 = dx_i * fy - dy_i * fx

            if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
                continue

            dx_j = bx[j_idx] - ax[j_idx]
            dy_j = by[j_idx] - ay[j_idx]
            gx = ax_i - ax[j_idx]
            gy = ay_i - ay[j_idx]
            hx = bx[i] - ax[j_idx]
            hy = by[i] - ay[j_idx]

            d3 = dx_j * gy - dy_j * gx
            d4 = dx_j * hy - dy_j * hx

            if (d3 > 0 and d4 > 0) or (d3 < 0 and d4 < 0):
                continue

            count += 1

    free(ax)
    free(ay)
    free(bx)
    free(by)
    return count
