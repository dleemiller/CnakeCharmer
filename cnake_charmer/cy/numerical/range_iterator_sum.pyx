# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum values from a cdef class iterator with __iter__ and __next__.

Keywords: iterator, cdef class, __iter__, __next__, range, sum, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class FilteredRange:
    """Range-like iterator that yields values where hash < threshold."""
    cdef long long current
    cdef long long stop
    cdef int step
    cdef int threshold

    def __cinit__(self, long long stop, int step, int threshold):
        self.current = 0
        self.stop = stop
        self.step = step
        self.threshold = threshold

    def __iter__(self):
        return self

    def __next__(self):
        cdef unsigned int h
        while self.current < self.stop:
            h = ((<unsigned long long>self.current * <unsigned long long>2654435761 + 7) >> 8) & 0xFF
            if h < self.threshold:
                val = self.current
                self.current += self.step
                return val
            self.current += self.step
        raise StopIteration


@cython_benchmark(syntax="cy", args=(100000,))
def range_iterator_sum(int n):
    """Sum values from a FilteredRange cdef class iterator."""
    cdef FilteredRange it = FilteredRange(n, 3, 128)
    cdef double total = 0.0
    cdef int count = 0
    cdef long long val

    for val in it:
        total += val * (1.0 + (count % 10) * 0.1)
        count += 1

    return total
