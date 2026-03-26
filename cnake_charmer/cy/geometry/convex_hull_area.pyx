# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute the area of the convex hull of n 2D points (Cython-optimized).

Keywords: convex hull, graham scan, shoelace, geometry, area, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, atan2
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def convex_hull_area(int n):
    """Compute the area of the convex hull of n deterministic 2D points."""
    cdef int i, j, m, pivot, top
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    cdef double *angles = <double *>malloc(n * sizeof(double))
    cdef int *indices = <int *>malloc(n * sizeof(int))
    cdef int *stack = <int *>malloc(n * sizeof(int))
    if not xs or not ys or not angles or not indices or not stack:
        free(xs); free(ys); free(angles); free(indices); free(stack)
        raise MemoryError()

    cdef double tmp_x, tmp_y, px, py, cross, area, tmp_a
    cdef int tmp_idx

    # Generate points
    for i in range(n):
        xs[i] = sin(i * 0.7) * 100.0
        ys[i] = cos(i * 1.3) * 100.0

    # Find bottom-most point
    pivot = 0
    for i in range(1, n):
        if ys[i] < ys[pivot] or (ys[i] == ys[pivot] and xs[i] < xs[pivot]):
            pivot = i

    # Swap pivot to index 0
    tmp_x = xs[0]; tmp_y = ys[0]
    xs[0] = xs[pivot]; ys[0] = ys[pivot]
    xs[pivot] = tmp_x; ys[pivot] = tmp_y

    px = xs[0]
    py = ys[0]

    # Compute angles for all other points
    cdef int count = n - 1
    for i in range(count):
        indices[i] = i + 1
        angles[i] = atan2(ys[i + 1] - py, xs[i + 1] - px)

    # Simple insertion sort by angle (then by distance for ties)
    cdef double dist_a, dist_b
    for i in range(1, count):
        tmp_a = angles[i]
        tmp_idx = indices[i]
        j = i - 1
        while j >= 0:
            if angles[j] > tmp_a or (angles[j] == tmp_a and ((xs[indices[j]] - px) * (xs[indices[j]] - px) + (ys[indices[j]] - py) * (ys[indices[j]] - py)) > ((xs[tmp_idx] - px) * (xs[tmp_idx] - px) + (ys[tmp_idx] - py) * (ys[tmp_idx] - py))):
                angles[j + 1] = angles[j]
                indices[j + 1] = indices[j]
                j -= 1
            else:
                break
        angles[j + 1] = tmp_a
        indices[j + 1] = tmp_idx

    # Graham scan
    stack[0] = 0
    top = 1
    for i in range(count):
        while top > 1:
            cross = (xs[stack[top - 1]] - xs[stack[top - 2]]) * (ys[indices[i]] - ys[stack[top - 2]]) - \
                    (ys[stack[top - 1]] - ys[stack[top - 2]]) * (xs[indices[i]] - xs[stack[top - 2]])
            if cross <= 0:
                top -= 1
            else:
                break
        stack[top] = indices[i]
        top += 1

    # Shoelace formula
    area = 0.0
    for i in range(top):
        j = (i + 1) % top
        area += xs[stack[i]] * ys[stack[j]]
        area -= xs[stack[j]] * ys[stack[i]]
    if area < 0:
        area = -area
    area = area / 2.0

    free(xs); free(ys); free(angles); free(indices); free(stack)
    return area
