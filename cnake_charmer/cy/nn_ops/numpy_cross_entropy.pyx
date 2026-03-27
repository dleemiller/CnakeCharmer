# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cross-entropy loss with fused loop (Cython).

Single loop avoids NumPy temporaries for log, multiply, add.

Keywords: nn_ops, cross entropy, loss, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, log
from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(100000,))
def numpy_cross_entropy(int n):
    """Compute cross-entropy loss, return scalar value."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[double, ndim=1] raw_arr = rng.standard_normal(n)
    cdef double[::1] raw = raw_arr

    cdef cnp.ndarray[double, ndim=1] labels_arr = (
        (rng.random(n) > 0.5).astype(np.float64)
    )
    cdef double[::1] labels = labels_arr

    cdef double loss = 0.0
    cdef double p, lbl
    cdef double eps = 1e-12
    cdef int i

    with nogil:
        for i in range(n):
            p = 1.0 / (1.0 + exp(-raw[i]))
            if p < eps:
                p = eps
            elif p > 1.0 - eps:
                p = 1.0 - eps
            lbl = labels[i]
            loss -= lbl * log(p) + (1.0 - lbl) * log(1.0 - p)

    return loss
