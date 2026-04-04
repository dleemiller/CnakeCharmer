# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count how many test points lie inside a regular 12-gon using ray casting (Cython).

Keywords: point in polygon, ray casting, geometry, containment, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport cos, sin, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def point_in_polygon(int n):
    """Count how many of n test points lie inside a regular 12-gon."""
    cdef int num_verts = 12
    cdef double *poly_x = <double *>malloc(num_verts * sizeof(double))
    cdef double *poly_y = <double *>malloc(num_verts * sizeof(double))
    if not poly_x or not poly_y:
        free(poly_x); free(poly_y)
        raise MemoryError()

    cdef int i, k, j_idx
    cdef double tx, ty, yi, yj, xi, xj
    cdef int inside
    cdef int count = 0
    cdef int first_inside_idx = -1
    cdef int last_inside_idx = -1

    # Build 12-gon
    for k in range(num_verts):
        poly_x[k] = 50.0 * cos(2.0 * M_PI * k / num_verts)
        poly_y[k] = 50.0 * sin(2.0 * M_PI * k / num_verts)

    for i in range(n):
        tx = <double>((i * 17 + 3) % 200 - 100)
        ty = <double>((i * 13 + 7) % 200 - 100)

        inside = 0
        j_idx = num_verts - 1
        for k in range(num_verts):
            yi = poly_y[k]
            yj = poly_y[j_idx]
            xi = poly_x[k]
            xj = poly_x[j_idx]

            if ((yi > ty) != (yj > ty)) and (tx < (xj - xi) * (ty - yi) / (yj - yi) + xi):
                inside = 1 - inside
            j_idx = k

        if inside:
            count += 1
            if first_inside_idx == -1:
                first_inside_idx = i
            last_inside_idx = i

    free(poly_x)
    free(poly_y)
    return (count, first_inside_idx, last_inside_idx)
