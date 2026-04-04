# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Matrix trace via buffer-backed 2D double array.

Fills a malloc'd double array, exposes it as a 2D buffer with
shape and strides via the buffer protocol on a cdef class,
obtains a 2D typed memoryview, and computes the trace.

Keywords: numerical, buffer protocol, matrix, trace, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cpython.buffer cimport Py_buffer
from cnake_data.benchmarks import cython_benchmark


cdef class MatrixBuffer:
    cdef double *data
    cdef Py_ssize_t rows
    cdef Py_ssize_t cols
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _strides[2]

    def __cinit__(self, Py_ssize_t rows, Py_ssize_t cols):
        self.rows = rows
        self.cols = cols
        self.data = <double *>malloc(
            rows * cols * sizeof(double)
        )
        if not self.data:
            raise MemoryError()
        self._shape[0] = rows
        self._shape[1] = cols
        self._strides[0] = cols * sizeof(double)
        self._strides[1] = sizeof(double)

    def __dealloc__(self):
        if self.data:
            free(self.data)

    def __getbuffer__(self, Py_buffer *view, int flags):
        view.buf = <void *>self.data
        view.len = (
            self.rows * self.cols * sizeof(double)
        )
        view.itemsize = sizeof(double)
        view.format = "d"
        view.ndim = 2
        view.shape = self._shape
        view.strides = self._strides
        view.suboffsets = NULL
        view.obj = self
        view.readonly = 0

    def __releasebuffer__(self, Py_buffer *view):
        pass


@cython_benchmark(syntax="cy", args=(200,))
def buffer_matrix_trace(int n):
    """Compute trace of hash-filled n x n matrix."""
    cdef unsigned int h
    cdef int i, j, idx
    cdef MatrixBuffer buf = MatrixBuffer(n, n)

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            h = (
                (<unsigned int>idx
                 * <unsigned int>2654435761)
                ^ (<unsigned int>idx
                   * <unsigned int>2246822519)
            )
            buf.data[idx] = (
                <double>(h & 0xFFFF) / 65535.0
            )

    cdef double[:, :] view = buf
    cdef double trace = 0.0
    for i in range(n):
        trace += view[i, i]
    return trace
