# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute the area of the convex hull of n 2D points (Cython-optimized).

Keywords: convex hull, graham scan, shoelace, geometry, area, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport sin, cos, atan2
from cnake_charmer.benchmarks import cython_benchmark


cdef struct PointAngle:
    double angle
    double dist
    int idx


cdef int _compare_points(const void *a, const void *b) noexcept nogil:
    """Compare two PointAngle structs for qsort."""
    cdef PointAngle *pa = <PointAngle *>a
    cdef PointAngle *pb = <PointAngle *>b
    if pa.angle < pb.angle:
        return -1
    elif pa.angle > pb.angle:
        return 1
    elif pa.dist < pb.dist:
        return -1
    elif pa.dist > pb.dist:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(50000,))
def convex_hull_area(int n):
    """Compute the area of the convex hull of n deterministic 2D points."""
    cdef int i, j, pivot, top
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    cdef PointAngle *pa = <PointAngle *>malloc((n - 1) * sizeof(PointAngle))
    cdef int *stack = <int *>malloc(n * sizeof(int))
    if not xs or not ys or not pa or not stack:
        free(xs); free(ys); free(pa); free(stack)
        raise MemoryError()

    cdef double tmp_x, tmp_y, px, py, cross, area

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

    # Build angle array for C-level qsort
    for i in range(n - 1):
        j = i + 1
        pa[i].angle = atan2(ys[j] - py, xs[j] - px)
        pa[i].dist = (xs[j] - px) * (xs[j] - px) + (ys[j] - py) * (ys[j] - py)
        pa[i].idx = j

    qsort(pa, n - 1, sizeof(PointAngle), _compare_points)

    # Graham scan
    stack[0] = 0
    top = 1
    cdef int idx
    for i in range(n - 1):
        idx = pa[i].idx
        while top > 1:
            cross = (xs[stack[top - 1]] - xs[stack[top - 2]]) * (ys[idx] - ys[stack[top - 2]]) - \
                    (ys[stack[top - 1]] - ys[stack[top - 2]]) * (xs[idx] - xs[stack[top - 2]])
            if cross <= 0:
                top -= 1
            else:
                break
        stack[top] = idx
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

    free(xs); free(ys); free(pa); free(stack)
    return area
