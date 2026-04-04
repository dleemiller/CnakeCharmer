# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Greedy circle packing density in a unit square (Cython-optimized).

Keywords: geometry, circle packing, greedy, density, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def circle_packing(int n):
    """Greedily pack circles into a unit square and return packing statistics."""
    cdef double *cx_list = <double *>malloc(n * sizeof(double))
    cdef double *cy_list = <double *>malloc(n * sizeof(double))
    cdef double *cr_list = <double *>malloc(n * sizeof(double))
    cdef double *accepted_x = <double *>malloc(n * sizeof(double))
    cdef double *accepted_y = <double *>malloc(n * sizeof(double))
    cdef double *accepted_r = <double *>malloc(n * sizeof(double))
    if not cx_list or not cy_list or not cr_list or not accepted_x or not accepted_y or not accepted_r:
        free(cx_list); free(cy_list); free(cr_list)
        free(accepted_x); free(accepted_y); free(accepted_r)
        raise MemoryError()

    cdef int i, j, count, overlap
    cdef double x, y, r, dx, dy, min_dist, total_area

    # Pre-generate candidates
    for i in range(n):
        cx_list[i] = ((i * 7 + 13) * 0.6180339887) % 1.0
        cy_list[i] = ((i * 11 + 17) * 0.4142135624) % 1.0
        cr_list[i] = 0.005 + 0.04 * ((i * 3 + 7) % 50) / 50.0

    count = 0
    total_area = 0.0

    for i in range(n):
        x = cx_list[i]
        y = cy_list[i]
        r = cr_list[i]

        # Check if circle fits inside unit square
        if x - r < 0.0 or x + r > 1.0 or y - r < 0.0 or y + r > 1.0:
            continue

        # Check overlap with all accepted circles
        overlap = 0
        for j in range(count):
            dx = x - accepted_x[j]
            dy = y - accepted_y[j]
            min_dist = r + accepted_r[j]
            if dx * dx + dy * dy < min_dist * min_dist:
                overlap = 1
                break

        if not overlap:
            accepted_x[count] = x
            accepted_y[count] = y
            accepted_r[count] = r
            count += 1
            total_area += M_PI * r * r

    free(cx_list); free(cy_list); free(cr_list)
    free(accepted_x); free(accepted_y); free(accepted_r)
    return (total_area, count)
