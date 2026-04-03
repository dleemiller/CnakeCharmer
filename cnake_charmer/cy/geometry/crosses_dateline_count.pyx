# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count synthetic line segments that cross the dateline (Cython)."""

from libc.stdlib cimport abs

from cnake_charmer.benchmarks import cython_benchmark


cdef inline bint _crosses(double x0, double x1) noexcept:
    if x0 == x1:
        return False
    if not ((x0 < 0.0 < x1) or (x1 < 0.0 < x0)):
        return False
    return abs(<int>(x1 - x0)) > 180


@cython_benchmark(syntax="cy", args=(7, 200000))
def crosses_dateline_count(int seed, int segment_count):
    cdef int i
    cdef int count = 0
    cdef int checksum = 0
    cdef unsigned int state = <unsigned int>seed
    cdef double x0, x1
    for i in range(segment_count):
        state = 1103515245u * state + 12345u
        x0 = ((state >> 8) % 360u) - 180.0
        state = 1103515245u * state + 12345u
        x1 = ((state >> 8) % 360u) - 180.0
        if _crosses(x0, x1):
            count += 1
            checksum += <int>((x1 - x0) * 10.0)
    return (count, checksum)

