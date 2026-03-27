# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run-length encoding of a 1D int array (Cython with memoryview).

Uses typed int memoryview for fast sequential run counting.

Keywords: string_processing, run length, encoding, numpy,
    typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(500000,))
def numpy_run_length(int n):
    """Count runs of consecutive equal values."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[int, ndim=1] data_arr = (
        rng.randint(0, 10, size=n).astype(np.intc)
    )
    cdef int[::1] data = data_arr
    cdef int runs = 1
    cdef int i

    with nogil:
        for i in range(1, n):
            if data[i] != data[i - 1]:
                runs += 1

    return runs
