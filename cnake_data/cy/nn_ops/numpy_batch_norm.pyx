# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batch normalization with typed memoryview (Cython).

Three-pass algorithm: mean, variance, then fused normalize+sum,
avoiding all NumPy temporary arrays.

Keywords: nn_ops, batch norm, normalization, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(100000,))
def numpy_batch_norm(int n):
    """Apply batch normalization and return sum."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[double, ndim=1] data_arr = rng.standard_normal(n)
    cdef double[::1] data = data_arr
    cdef double gamma = 1.5
    cdef double beta = 0.5
    cdef double eps = 1e-5
    cdef double mean = 0.0
    cdef double var = 0.0
    cdef double diff, total, scale
    cdef int i

    with nogil:
        for i in range(n):
            mean += data[i]
        mean /= n

        for i in range(n):
            diff = data[i] - mean
            var += diff * diff
        var /= n

        scale = gamma / sqrt(var + eps)
        total = 0.0
        for i in range(n):
            total += (data[i] - mean) * scale + beta

    return total
