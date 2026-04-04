# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum coordinates of rapidly created/destroyed Point objects.

Keywords: geometry, point, freelist, extension type, cython, benchmark
"""

cimport cython
from cnake_data.benchmarks import cython_benchmark


@cython.freelist(64)
cdef class Point:
    """Simple 2D point with freelist optimization."""
    cdef double x
    cdef double y

    def __cinit__(self, double x, double y):
        self.x = x
        self.y = y


@cython_benchmark(syntax="cy", args=(100000,))
def freelist_point_sum(int n):
    """Create n points, sum all x and y coordinates."""
    cdef double total = 0.0
    cdef int i
    cdef double x, y
    cdef Point p

    for i in range(n):
        x = ((<long long>i * <long long>2654435761)
             % 100000) / 100.0
        y = ((<long long>i * <long long>1664525
              + <long long>1013904223) % 100000) / 100.0
        p = Point(x, y)
        total += p.x + p.y
    return total
