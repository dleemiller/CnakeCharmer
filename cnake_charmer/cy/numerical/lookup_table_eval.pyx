# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate piecewise linear function via cdef class lookup table.

Keywords: lookup table, cdef class, __getitem__, __len__, interpolation, cython, benchmark
"""

from libc.math cimport sin, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef class LookupTable:
    """Fixed-size lookup table with __getitem__ and __len__."""
    cdef double *data
    cdef Py_ssize_t _size

    def __cinit__(self, Py_ssize_t size):
        self._size = size
        self.data = <double *>malloc(size * sizeof(double))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            free(self.data)

    def __len__(self):
        return self._size

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self._size:
            raise IndexError("index out of range")
        return self.data[idx]

    cdef inline double get_fast(self, int idx):
        return self.data[idx]

    cdef double interpolate(self, double frac_idx):
        """Linearly interpolate between adjacent entries."""
        cdef int idx = <int>frac_idx
        cdef double frac = frac_idx - idx
        if idx >= self._size - 1:
            return self.data[self._size - 1]
        return self.data[idx] * (1.0 - frac) + self.data[idx + 1] * frac


@cython_benchmark(syntax="cy", args=(100000,))
def lookup_table_eval(int n):
    """Build a LookupTable and evaluate interpolated lookups."""
    cdef int table_size = 256
    cdef LookupTable table = LookupTable(table_size)
    cdef int i

    for i in range(table_size):
        table.data[i] = sin(2.0 * M_PI * i / table_size) * 100.0

    cdef double total = 0.0
    cdef unsigned long long h
    cdef double frac_idx

    for i in range(n):
        h = ((<unsigned long long>i * <unsigned long long>2654435761 + 17) >> 4) & 0xFFFF
        frac_idx = (h % 25500) / 100.0

        total += table.interpolate(frac_idx)

    return total
