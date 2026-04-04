# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Shoelace polygon area and point-in-polygon count for a regular n-gon (Cython-optimized).

Keywords: geometry, polygon, winding, shoelace, point-in-polygon, ray casting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport cos, sin, fabs
from cnake_charmer.benchmarks import cython_benchmark

cdef double M_PI = 3.141592653589793238462643383


@cython_benchmark(syntax="cy", args=(5000,))
def polygon_winding(int n):
    """Compute area of a regular n-gon and count how many test points are inside.

    Args:
        n: Number of polygon vertices (and test points).

    Returns:
        Tuple of (area_times_1e6_as_int, inside_count).
    """
    cdef int k, kn, j, i
    cdef double two_pi, area, px, py
    cdef double xi, yi, xj, yj
    cdef int inside_count = 0
    cdef int inside
    cdef double *vx = <double *>malloc(n * sizeof(double))
    cdef double *vy = <double *>malloc(n * sizeof(double))
    if not vx or not vy:
        if vx:
            free(vx)
        if vy:
            free(vy)
        raise MemoryError()

    two_pi = 2.0 * M_PI

    with nogil:
        # Build vertices
        for k in range(n):
            vx[k] = cos(two_pi * k / n)
            vy[k] = sin(two_pi * k / n)

        # Shoelace area
        area = 0.0
        for k in range(n):
            kn = (k + 1) % n
            area += vx[k] * vy[kn] - vx[kn] * vy[k]
        area = fabs(area) * 0.5

        # Ray casting for each test point
        for i in range(n):
            px = (i * 0.618 - <int>(i * 0.618)) * 2.0 - 1.0
            py = (i * 0.382 - <int>(i * 0.382)) * 2.0 - 1.0
            inside = 0
            j = n - 1
            for k in range(n):
                xi = vx[k]
                yi = vy[k]
                xj = vx[j]
                yj = vy[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                    inside = 1 - inside
                j = k
            if inside:
                inside_count += 1

    free(vx)
    free(vy)
    return (<long long>(area * 1e6 + 0.5), inside_count)
