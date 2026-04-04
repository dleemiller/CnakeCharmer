# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gift wrapping (Jarvis march) convex hull algorithm (Cython-optimized).

Keywords: geometry, convex hull, gift wrapping, jarvis march, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def gift_wrapping(int n):
    """Compute convex hull via gift wrapping and return number of hull vertices."""
    cdef int i, start, current, candidate, hull_count
    cdef double cross, d_cand, d_i
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    if not xs or not ys:
        free(xs)
        free(ys)
        raise MemoryError()

    # Generate points
    for i in range(n):
        xs[i] = sin(i * 0.7) * 100.0
        ys[i] = cos(i * 1.3) * 100.0

    # Find leftmost point
    start = 0
    for i in range(1, n):
        if xs[i] < xs[start] or (xs[i] == xs[start] and ys[i] < ys[start]):
            start = i

    hull_count = 0
    current = start
    while True:
        hull_count += 1
        candidate = 0
        for i in range(1, n):
            if i == current:
                continue
            if candidate == current:
                candidate = i
                continue
            cross = ((xs[candidate] - xs[current]) * (ys[i] - ys[current]) -
                     (ys[candidate] - ys[current]) * (xs[i] - xs[current]))
            if cross < 0:
                candidate = i
            elif cross == 0:
                d_cand = ((xs[candidate] - xs[current]) * (xs[candidate] - xs[current]) +
                          (ys[candidate] - ys[current]) * (ys[candidate] - ys[current]))
                d_i = ((xs[i] - xs[current]) * (xs[i] - xs[current]) +
                       (ys[i] - ys[current]) * (ys[i] - ys[current]))
                if d_i > d_cand:
                    candidate = i

        current = candidate
        if current == start:
            break
        if hull_count > n:
            break

    free(xs)
    free(ys)
    return hull_count
