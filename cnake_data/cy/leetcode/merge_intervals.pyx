# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Merge overlapping intervals and return the count of merged intervals.

Keywords: leetcode, merge intervals, sorting, sweep, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_data.benchmarks import cython_benchmark


cdef struct Interval:
    int start
    int end


cdef int cmp_interval(const void *a, const void *b) noexcept nogil:
    cdef Interval *ia = <Interval *>a
    cdef Interval *ib = <Interval *>b
    if ia.start != ib.start:
        return ia.start - ib.start
    return ia.end - ib.end


@cython_benchmark(syntax="cy", args=(1000000,))
def merge_intervals(int n):
    """Generate n intervals, merge overlapping ones, return merged count."""
    cdef Interval *intervals = <Interval *>malloc(n * sizeof(Interval))
    if not intervals:
        raise MemoryError()

    cdef int i, merged_count
    cdef int cur_start, cur_end

    # Generate intervals
    for i in range(n):
        intervals[i].start = (i * 3) % 1000
        intervals[i].end = (i * 3) % 1000 + (i % 20 + 1)

    # Sort
    qsort(intervals, n, sizeof(Interval), cmp_interval)

    # Merge
    merged_count = 0
    cur_start = intervals[0].start
    cur_end = intervals[0].end

    for i in range(1, n):
        if intervals[i].start <= cur_end:
            if intervals[i].end > cur_end:
                cur_end = intervals[i].end
        else:
            merged_count += 1
            cur_start = intervals[i].start
            cur_end = intervals[i].end

    merged_count += 1

    free(intervals)
    return merged_count
