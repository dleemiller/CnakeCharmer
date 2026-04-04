# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Sum of areas of n deterministic 8-sided polygons using the shoelace formula (Cython-optimized).

Keywords: geometry, polygon, area, shoelace, cython, benchmark
"""

from libc.math cimport cos, sin, fabs, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def polygon_area(int n):
    """Compute the total area of n polygons using the shoelace formula with typed loops.

    Args:
        n: Number of polygons.

    Returns:
        Sum of all polygon areas.
    """
    cdef int i, j, j_next
    cdef double total_area = 0.0
    cdef double area, angle, r
    cdef double two_pi_over_8 = 2.0 * M_PI / 8.0
    cdef double vx[8]
    cdef double vy[8]

    for i in range(n):
        for j in range(8):
            r = (j * i + 3) % 50 + 10
            angle = j * two_pi_over_8
            vx[j] = r * cos(angle)
            vy[j] = r * sin(angle)

        area = 0.0
        for j in range(8):
            j_next = (j + 1) % 8
            area += vx[j] * vy[j_next] - vx[j_next] * vy[j]
        total_area += fabs(area) * 0.5

    return total_area
