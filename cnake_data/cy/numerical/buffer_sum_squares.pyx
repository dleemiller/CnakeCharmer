# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum of squares via buffer-backed double array.

Fills a malloc'd double array, exposes it through the buffer
protocol on a cdef class, obtains a typed memoryview, and
computes the sum of squares.

Keywords: numerical, buffer protocol, sum of squares, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cpython.buffer cimport Py_buffer
from cnake_data.benchmarks import cython_benchmark


cdef class DoubleBuffer:
    cdef double *data
    cdef Py_ssize_t length
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]

    def __cinit__(self, Py_ssize_t n):
        self.length = n
        self.data = <double *>malloc(
            n * sizeof(double)
        )
        if not self.data:
            raise MemoryError()
        self._shape[0] = n
        self._strides[0] = sizeof(double)

    def __dealloc__(self):
        if self.data:
            free(self.data)

    def __getbuffer__(self, Py_buffer *view, int flags):
        view.buf = <void *>self.data
        view.len = self.length * sizeof(double)
        view.itemsize = sizeof(double)
        view.format = "d"
        view.ndim = 1
        view.shape = self._shape
        view.strides = self._strides
        view.suboffsets = NULL
        view.obj = self
        view.readonly = 0

    def __releasebuffer__(self, Py_buffer *view):
        pass


@cython_benchmark(syntax="cy", args=(100000,))
def buffer_sum_squares(int n):
    """Compute sum of squares using buffer protocol."""
    cdef unsigned int h
    cdef int i
    cdef DoubleBuffer buf = DoubleBuffer(n)

    for i in range(n):
        h = (
            (<unsigned int>i * <unsigned int>2654435761)
            ^ (<unsigned int>i * <unsigned int>2246822519)
        )
        buf.data[i] = <double>(h & 0xFFFF) / 65535.0

    cdef double[:] view = buf
    cdef double total = 0.0
    for i in range(n):
        total += view[i] * view[i]
    return total
