# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Byte histogram via buffer-backed unsigned char array.

Fills a malloc'd unsigned char array, exposes it through the
buffer protocol, obtains a typed memoryview, and builds a
256-bin histogram returning the max bin count.

Keywords: algorithms, buffer protocol, histogram, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cpython.buffer cimport Py_buffer
from cnake_data.benchmarks import cython_benchmark


cdef class ByteBuffer:
    cdef unsigned char *data
    cdef Py_ssize_t length
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]

    def __cinit__(self, Py_ssize_t n):
        self.length = n
        self.data = <unsigned char *>malloc(
            n * sizeof(unsigned char)
        )
        if not self.data:
            raise MemoryError()
        self._shape[0] = n
        self._strides[0] = sizeof(unsigned char)

    def __dealloc__(self):
        if self.data:
            free(self.data)

    def __getbuffer__(self, Py_buffer *view, int flags):
        view.buf = <void *>self.data
        view.len = self.length * sizeof(unsigned char)
        view.itemsize = sizeof(unsigned char)
        view.format = "B"
        view.ndim = 1
        view.shape = self._shape
        view.strides = self._strides
        view.suboffsets = NULL
        view.obj = self
        view.readonly = 0

    def __releasebuffer__(self, Py_buffer *view):
        pass


@cython_benchmark(syntax="cy", args=(100000,))
def buffer_byte_histogram(int n):
    """Build byte histogram and return max bin count."""
    cdef unsigned int h
    cdef int i
    cdef ByteBuffer buf = ByteBuffer(n)

    for i in range(n):
        h = (
            (<unsigned int>i * <unsigned int>2654435761)
            ^ (<unsigned int>i * <unsigned int>2246822519)
        )
        buf.data[i] = (h >> 8) & 0xFF

    cdef unsigned char[:] view = buf
    cdef int histogram[256]
    memset(histogram, 0, 256 * sizeof(int))

    for i in range(n):
        histogram[view[i]] += 1

    cdef int max_count = 0
    for i in range(256):
        if histogram[i] > max_count:
            max_count = histogram[i]
    return max_count
