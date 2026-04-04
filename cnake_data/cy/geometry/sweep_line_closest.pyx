# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find closest pair distance using sweep line algorithm (Cython-optimized).

Keywords: closest pair, sweep line, geometry, computational geometry, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport sin, cos, sqrt
from cnake_data.benchmarks import cython_benchmark


cdef struct IndexedX:
    double x
    int idx


cdef int compare_x(const void *a, const void *b) noexcept nogil:
    cdef double ax = (<IndexedX *>a).x
    cdef double bx = (<IndexedX *>b).x
    if ax < bx:
        return -1
    elif ax > bx:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(50000,))
def sweep_line_closest(int n):
    """Find closest pair distance using sweep line with C arrays."""
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    cdef IndexedX *sorted_pts = <IndexedX *>malloc(n * sizeof(IndexedX))

    if not xs or not ys or not sorted_pts:
        if xs: free(xs)
        if ys: free(ys)
        if sorted_pts: free(sorted_pts)
        raise MemoryError()

    cdef int i, j, pi, pj
    cdef double xi, yi, dx, dy, dist_sq, best

    for i in range(n):
        xs[i] = sin(i * 0.7) * 1000.0
        ys[i] = cos(i * 1.3) * 1000.0
        sorted_pts[i].x = xs[i]
        sorted_pts[i].idx = i

    qsort(sorted_pts, n, sizeof(IndexedX), compare_x)

    best = 1e18
    for i in range(n):
        pi = sorted_pts[i].idx
        xi = xs[pi]
        yi = ys[pi]
        for j in range(i + 1, n):
            pj = sorted_pts[j].idx
            dx = xs[pj] - xi
            if dx * dx >= best:
                break
            dy = ys[pj] - yi
            dist_sq = dx * dx + dy * dy
            if dist_sq < best:
                best = dist_sq

    free(xs)
    free(ys)
    free(sorted_pts)
    return sqrt(best)
