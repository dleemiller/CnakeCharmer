# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pairwise 2D Euclidean distance accumulation (Cython).

Keywords: numerical, euclidean distance, geometry, cython, benchmark
"""

from libc.math cimport sqrt

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(17, 40000, 0.01))
def euclid_distance_pair(int seed, int pair_count, double scale):
    return _euclid_distance_pair_impl(seed, pair_count, scale)


cdef double _euclid_distance_pair_impl(int seed, int pair_count, double scale) noexcept:
    cdef int i
    cdef double total = 0.0
    cdef double x1, y1, x2, y2, dx, dy
    cdef unsigned int state = <unsigned int>seed
    for i in range(pair_count):
        state = 1664525u * state + 1013904223u
        x1 = ((state >> 8) & 0xFFFFu) * scale
        state = 1664525u * state + 1013904223u
        y1 = ((state >> 8) & 0xFFFFu) * scale
        state = 1664525u * state + 1013904223u
        x2 = ((state >> 8) & 0xFFFFu) * scale
        state = 1664525u * state + 1013904223u
        y2 = ((state >> 8) & 0xFFFFu) * scale
        dx = x1 - x2
        dy = y1 - y2
        total += sqrt(dx * dx + dy * dy)
    return total
