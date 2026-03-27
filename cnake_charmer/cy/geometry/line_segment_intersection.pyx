# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count intersections among n line segments with coordinate tracking (Cython).

Keywords: geometry, line segment, intersection, cross product, coordinate, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def line_segment_intersection(int n):
    """Count intersections among n line segments with coordinate tracking."""
    cdef double *ax = <double *>malloc(n * sizeof(double))
    cdef double *ay = <double *>malloc(n * sizeof(double))
    cdef double *bx = <double *>malloc(n * sizeof(double))
    cdef double *by = <double *>malloc(n * sizeof(double))
    if not ax or not ay or not bx or not by:
        free(ax); free(ay); free(bx); free(by)
        raise MemoryError()

    cdef int i, j
    cdef double ax_i, ay_i, dx_i, dy_i, dx_j, dy_j
    cdef double ex, ey, fx, fy, gx, gy, hx, hy
    cdef double d1, d2, d3, d4, denom, t, ix, iy
    cdef int count = 0
    cdef double first_ix = 0.0
    cdef double last_iy = 0.0

    for i in range(n):
        ax[i] = ((i * 73 + 11) % 997) * 0.1
        ay[i] = ((i * 37 + 23) % 991) * 0.1
        bx[i] = ((i * 53 + 7) % 983) * 0.1
        by[i] = ((i * 97 + 13) % 977) * 0.1

    for i in range(n):
        ax_i = ax[i]
        ay_i = ay[i]
        dx_i = bx[i] - ax_i
        dy_i = by[i] - ay_i
        for j in range(i + 1, n):
            ex = ax[j] - ax_i
            ey = ay[j] - ay_i
            fx = bx[j] - ax_i
            fy = by[j] - ay_i

            d1 = dx_i * ey - dy_i * ex
            d2 = dx_i * fy - dy_i * fx

            if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
                continue

            dx_j = bx[j] - ax[j]
            dy_j = by[j] - ay[j]
            gx = ax_i - ax[j]
            gy = ay_i - ay[j]
            hx = bx[i] - ax[j]
            hy = by[i] - ay[j]

            d3 = dx_j * gy - dy_j * gx
            d4 = dx_j * hy - dy_j * hx

            if (d3 > 0 and d4 > 0) or (d3 < 0 and d4 < 0):
                continue

            denom = dx_i * dy_j - dy_i * dx_j
            if denom == 0.0:
                count += 1
                continue

            t = ((ax[j] - ax_i) * dy_j - (ay[j] - ay_i) * dx_j) / denom
            ix = ax_i + t * dx_i
            iy = ay_i + t * dy_i

            count += 1
            if count == 1:
                first_ix = ix
            last_iy = iy

    free(ax)
    free(ay)
    free(bx)
    free(by)
    return (count, first_ix, last_iy)
