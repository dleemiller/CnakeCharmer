# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D median filter with window=5 (Cython with memoryview).

Uses typed memoryview and insertion sort for fast sliding
window median.

Keywords: dsp, median, filter, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_data.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(100000,))
def numpy_median_filter(int n):
    """Apply 1D median filter (window=5) and return sum."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[double, ndim=1] data_arr = (
        rng.standard_normal(n).astype(np.float64)
    )
    cdef double[::1] data = data_arr
    cdef cnp.ndarray[double, ndim=1] out_arr = (
        np.empty(n, dtype=np.float64)
    )
    cdef double[::1] out = out_arr
    cdef int i, j, k, lo, hi, wlen
    cdef double total = 0.0
    cdef double buf[5]
    cdef double tmp

    with nogil:
        for i in range(n):
            lo = i - 2
            if lo < 0:
                lo = 0
            hi = i + 3
            if hi > n:
                hi = n
            wlen = hi - lo
            for j in range(wlen):
                buf[j] = data[lo + j]
            for j in range(1, wlen):
                for k in range(j, 0, -1):
                    if buf[k - 1] > buf[k]:
                        tmp = buf[k]
                        buf[k] = buf[k - 1]
                        buf[k - 1] = tmp
                    else:
                        break
            out[i] = buf[wlen // 2]

        for i in range(n):
            total += out[i]

    return total
