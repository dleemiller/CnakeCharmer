# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute centroids of n deterministic polygons (Cython-optimized).

Keywords: geometry, polygon, centroid, signed area, cython, benchmark
"""

from libc.math cimport cos, sin, fabs, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def polygon_centroid(int n):
    """Compute the sum of centroid coordinates over n deterministic 6-sided polygons."""
    cdef int i, j, j1
    cdef double total_cx = 0.0
    cdef double total_cy = 0.0
    cdef double total_area = 0.0
    cdef double two_pi_over_6 = 2.0 * M_PI / 6.0
    cdef double area, cx, cy, cross, r, angle
    cdef double vx[6]
    cdef double vy[6]

    for i in range(n):
        for j in range(6):
            r = (j * i + 5) % 40 + 10
            angle = j * two_pi_over_6
            vx[j] = r * cos(angle)
            vy[j] = r * sin(angle)

        area = 0.0
        cx = 0.0
        cy = 0.0
        for j in range(6):
            j1 = (j + 1) % 6
            cross = vx[j] * vy[j1] - vx[j1] * vy[j]
            area += cross
            cx += (vx[j] + vx[j1]) * cross
            cy += (vy[j] + vy[j1]) * cross

        area *= 0.5
        if fabs(area) > 1e-12:
            cx = cx / (6.0 * area)
            cy = cy / (6.0 * area)
        else:
            cx = 0.0
            cy = 0.0

        total_cx += cx
        total_cy += cy
        total_area += fabs(area)

    return (total_cx, total_cy, total_area)
