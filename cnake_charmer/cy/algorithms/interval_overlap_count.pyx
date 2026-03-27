# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count overlapping interval pairs using cdef class with __richcmp__ and __hash__ (Cython).

Keywords: algorithms, intervals, overlap, comparison, hash, cdef class, cython, benchmark
"""

from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cnake_charmer.benchmarks import cython_benchmark


cdef class Interval:
    """Interval with start and end, comparable by start via __richcmp__."""
    cdef long long start
    cdef long long end

    def __cinit__(self, long long start, long long end):
        self.start = start
        self.end = end

    def __richcmp__(self, other, int op):
        cdef Interval o = <Interval>other
        if op == Py_LT:
            if self.start != o.start:
                return self.start < o.start
            return self.end < o.end
        elif op == Py_LE:
            if self.start != o.start:
                return self.start < o.start
            return self.end <= o.end
        elif op == Py_EQ:
            return self.start == o.start and self.end == o.end
        elif op == Py_NE:
            return self.start != o.start or self.end != o.end
        elif op == Py_GT:
            if self.start != o.start:
                return self.start > o.start
            return self.end > o.end
        elif op == Py_GE:
            if self.start != o.start:
                return self.start > o.start
            return self.end >= o.end
        return NotImplemented

    def __hash__(self):
        return <Py_hash_t>(self.start * <long long>100003 + self.end)


@cython_benchmark(syntax="cy", args=(20000,))
def interval_overlap_count(int n):
    """Create n intervals, sort by start, count overlapping pairs via sweep."""
    cdef list intervals = []
    cdef long long start, length
    cdef int i, j
    cdef long long overlap_count = 0
    cdef Interval iv

    for i in range(n):
        start = ((<long long>i * <long long>2654435761 + 17) ^ (<long long>i * <long long>1103515245)) % 10000
        length = ((<long long>i * <long long>1664525 + <long long>1013904223) ^ (<long long>i * <long long>214013)) % 500 + 1
        intervals.append(Interval(start, start + length))

    intervals.sort()

    # Sweep line: count pairs where previous interval end > current start
    cdef list active_ends = []
    cdef list new_active
    cdef long long e

    for i in range(n):
        iv = <Interval>intervals[i]
        new_active = []
        for j in range(len(active_ends)):
            e = <long long>active_ends[j]
            if e > iv.start:
                new_active.append(e)
        active_ends = new_active
        overlap_count += len(active_ends)
        active_ends.append(iv.end)

    return overlap_count
