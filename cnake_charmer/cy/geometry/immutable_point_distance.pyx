# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute total pairwise distances for immutable 2D points using cdef class (Cython).

Keywords: geometry, point, distance, immutable, readonly, cdef class, cython, benchmark
"""

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef class ImmutablePoint:
    """2D point with cdef readonly x and y coordinates."""
    cdef readonly double x
    cdef readonly double y

    def __cinit__(self, double x, double y):
        self.x = x
        self.y = y


@cython_benchmark(syntax="cy", args=(3000,))
def immutable_point_distance(int n):
    """Create n immutable points, compute sum of all pairwise distances."""
    cdef double *xs
    cdef double *ys
    cdef double total = 0.0
    cdef double dx, dy, px, py_
    cdef int i, j

    # Create points and extract coordinates for fast inner loop
    xs = <double *>malloc(n * sizeof(double))
    ys = <double *>malloc(n * sizeof(double))
    if not xs or not ys:
        if xs:
            free(xs)
        if ys:
            free(ys)
        raise MemoryError()

    for i in range(n):
        xs[i] = ((<long long>i * <long long>2654435761 + 17) % 10000) / 100.0
        ys[i] = ((<long long>i * <long long>1103515245 + 12345) % 10000) / 100.0

    for i in range(n):
        px = xs[i]
        py_ = ys[i]
        for j in range(i + 1, n):
            dx = px - xs[j]
            dy = py_ - ys[j]
            total += sqrt(dx * dx + dy * dy)

    free(xs)
    free(ys)

    return total
