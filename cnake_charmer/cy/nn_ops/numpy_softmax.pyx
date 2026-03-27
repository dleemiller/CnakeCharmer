# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Chunked softmax with typed memoryview (Cython).

Single-pass max, fused exp+sum, and divide passes avoid
NumPy temporary allocations.

Keywords: nn_ops, softmax, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport exp
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(100000,))
def numpy_softmax(int n):
    """Compute chunked softmax and return sum of outputs."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[double, ndim=1] data_arr = rng.standard_normal(n)
    cdef double[::1] data = data_arr
    cdef int chunk = 256
    cdef int num_chunks = n // chunk
    cdef double total = 0.0
    cdef double mx, s_exp, inv_s_exp
    cdef int c, i, start
    cdef double *buf = <double *>malloc(chunk * sizeof(double))
    if not buf:
        raise MemoryError()

    with nogil:
        for c in range(num_chunks):
            start = c * chunk
            mx = data[start]
            for i in range(1, chunk):
                if data[start + i] > mx:
                    mx = data[start + i]
            s_exp = 0.0
            for i in range(chunk):
                buf[i] = exp(data[start + i] - mx)
                s_exp += buf[i]
            inv_s_exp = 1.0 / s_exp
            for i in range(chunk):
                total += buf[i] * inv_s_exp

    free(buf)
    return total
