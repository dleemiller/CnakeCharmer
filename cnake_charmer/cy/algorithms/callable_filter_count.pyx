# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count values passing a threshold filter using cdef class with __call__ (Cython).

Keywords: algorithms, callable, filter, threshold, cdef class, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class ThresholdFilter:
    """Callable filter that returns True if value is within [lo, hi]."""
    cdef long long lo
    cdef long long hi

    def __cinit__(self, long long lo, long long hi):
        self.lo = lo
        self.hi = hi

    def __call__(self, long long x):
        return self.lo <= x <= self.hi

    cdef inline bint check(self, long long x):
        return self.lo <= x <= self.hi


@cython_benchmark(syntax="cy", args=(500000,))
def callable_filter_count(int n):
    """Apply threshold filters to n values, count how many pass."""
    cdef list filters = []
    cdef long long lo, hi, val
    cdef int i, count
    cdef ThresholdFilter f

    for i in range(8):
        lo = ((<long long>i * <long long>2654435761 + 17) % 500)
        hi = lo + ((<long long>i * <long long>1103515245 + 12345) % 500) + 1
        filters.append(ThresholdFilter(lo, hi))

    count = 0
    for i in range(n):
        val = ((<long long>i * <long long>1664525 + <long long>1013904223) ^ (<long long>i * <long long>214013 + <long long>2531011)) % 1000
        f = <ThresholdFilter>filters[i & 7]
        if f.check(val):
            count += 1

    return count
