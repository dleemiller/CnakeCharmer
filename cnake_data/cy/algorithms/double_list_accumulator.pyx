# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class wrapping paired numeric buffers with aggregation methods (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef class DoubleList:
    cdef int n
    cdef double *left
    cdef double *right

    def __cinit__(self, int n, double scale):
        cdef int i
        self.n = n
        self.left = <double *>malloc(n * sizeof(double))
        self.right = <double *>malloc(n * sizeof(double))
        if not self.left or not self.right:
            free(self.left)
            free(self.right)
            self.left = NULL
            self.right = NULL
            raise MemoryError()
        for i in range(n):
            self.left[i] = ((i * 17 + 3) % 100) * scale
            self.right[i] = ((i * 19 + 5) % 100) * scale

    def __dealloc__(self):
        if self.left != NULL:
            free(self.left)
        if self.right != NULL:
            free(self.right)


cdef void _run_blend(DoubleList obj, int rounds, double alpha, double beta, double *total_out, double *peak_out) noexcept nogil:
    cdef int r, i
    cdef double a, b, v, total = 0.0, peak = 0.0
    for r in range(rounds):
        a = alpha + (r & 3) * 0.01
        b = beta - (r & 1) * 0.02
        v = 0.0
        for i in range(obj.n):
            v += obj.left[i] * a + obj.right[i] * b
        total += v
        if v > peak:
            peak = v
    total_out[0] = total
    peak_out[0] = peak


@cython_benchmark(syntax="cy", args=(700, 0.17, 900, 0.61, 0.39))
def double_list_accumulator(int n, double scale, int rounds, double alpha, double beta):
    cdef DoubleList obj = DoubleList(n, scale)
    cdef double total = 0.0
    cdef double peak = 0.0
    with nogil:
        _run_blend(obj, rounds, alpha, beta, &total, &peak)
    return (total, peak, obj.left[n // 2])
