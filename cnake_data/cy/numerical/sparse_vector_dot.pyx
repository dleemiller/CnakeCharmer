# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dot product of sparse vectors using cdef class with __getitem__ and __len__.

Keywords: sparse vector, cdef class, __getitem__, __len__, dot product, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef class SparseVector:
    """Sparse vector stored as parallel index/value arrays."""
    cdef int *indices
    cdef double *values
    cdef Py_ssize_t _size
    cdef Py_ssize_t capacity

    def __cinit__(self, Py_ssize_t capacity):
        self.capacity = capacity
        self._size = 0
        self.indices = <int *>malloc(capacity * sizeof(int))
        self.values = <double *>malloc(capacity * sizeof(double))
        if not self.indices or not self.values:
            raise MemoryError()

    def __dealloc__(self):
        if self.indices:
            free(self.indices)
        if self.values:
            free(self.values)

    def __len__(self):
        return self._size

    def __getitem__(self, Py_ssize_t idx):
        """Return (index, value) pair at position idx."""
        if idx < 0 or idx >= self._size:
            raise IndexError("index out of range")
        return (self.indices[idx], self.values[idx])

    cdef void append(self, int idx, double val):
        self.indices[self._size] = idx
        self.values[self._size] = val
        self._size += 1

    cdef double dot(self, SparseVector other):
        """Compute dot product by merging sorted index lists."""
        cdef Py_ssize_t ia = 0, ib = 0
        cdef double result = 0.0
        while ia < self._size and ib < other._size:
            if self.indices[ia] == other.indices[ib]:
                result += self.values[ia] * other.values[ib]
                ia += 1
                ib += 1
            elif self.indices[ia] < other.indices[ib]:
                ia += 1
            else:
                ib += 1
        return result


@cython_benchmark(syntax="cy", args=(50000,))
def sparse_vector_dot(int n):
    """Create two SparseVector cdef classes and compute dot product."""
    cdef SparseVector a = SparseVector(n // 5)
    cdef SparseVector b = SparseVector(n // 5)
    cdef int i
    cdef unsigned int h
    cdef double val

    for i in range(n):
        h = ((<unsigned long long>i * <unsigned long long>2654435761 + 1) >> 8) & 0xFF
        if h < 26:
            val = ((<int>((h * 31 + 7) % 200)) - 100) / 10.0
            a.append(i, val)

    for i in range(n):
        h = ((<unsigned long long>i * <unsigned long long>1103515245 + 3) >> 8) & 0xFF
        if h < 26:
            val = ((<int>((h * 37 + 11) % 200)) - 100) / 10.0
            b.append(i, val)

    return a.dot(b)
