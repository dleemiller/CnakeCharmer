# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build histogram using cdef class with __getitem__, __setitem__, __len__.

Keywords: histogram, cdef class, __getitem__, __setitem__, __len__, statistics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport log
from cnake_data.benchmarks import cython_benchmark


cdef class Histogram:
    """Integer histogram with __getitem__, __setitem__, __len__."""
    cdef int *counts
    cdef Py_ssize_t num_bins

    def __cinit__(self, Py_ssize_t num_bins):
        self.num_bins = num_bins
        self.counts = <int *>malloc(num_bins * sizeof(int))
        if not self.counts:
            raise MemoryError()
        cdef Py_ssize_t i
        for i in range(num_bins):
            self.counts[i] = 0

    def __dealloc__(self):
        if self.counts:
            free(self.counts)

    def __len__(self):
        return self.num_bins

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.num_bins:
            raise IndexError("index out of range")
        return self.counts[idx]

    def __setitem__(self, Py_ssize_t idx, int val):
        if idx < 0 or idx >= self.num_bins:
            raise IndexError("index out of range")
        self.counts[idx] = val

    cdef inline void increment(self, int bin_idx):
        self.counts[bin_idx] += 1


@cython_benchmark(syntax="cy", args=(100000,))
def histogram_bucket(int n):
    """Build a Histogram cdef class and compute statistics."""
    cdef int num_bins = 256
    cdef Histogram hist = Histogram(num_bins)
    cdef int i
    cdef unsigned int h

    for i in range(n):
        h = ((<unsigned long long>i * <unsigned long long>2654435761 + 7) >> 4) & 0xFF
        hist.increment(h)

    cdef int max_count = 0, nonempty = 0, c
    for i in range(num_bins):
        c = hist.counts[i]
        if c > max_count:
            max_count = c
        if c > 0:
            nonempty += 1

    cdef double entropy = 0.0, p
    for i in range(num_bins):
        if hist.counts[i] > 0:
            p = <double>hist.counts[i] / n
            entropy -= p * log(p)

    return (max_count, nonempty, <int>(entropy * 1000))
