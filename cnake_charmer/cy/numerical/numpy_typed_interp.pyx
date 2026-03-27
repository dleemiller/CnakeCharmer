# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Linear interpolation with cnp.float64_t types (Cython).

Binary search + lerp using typed memoryview and cnp types.

Keywords: numerical, interpolation, linear, numpy, cnp,
    typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


cdef inline int bisect(
    cnp.float64_t[::1] arr, int length, cnp.float64_t val
) noexcept nogil:
    """Binary search for rightmost index where arr[i] <= val."""
    cdef int lo = 0
    cdef int hi = length - 1
    cdef int mid
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid - 1
    return lo


@cython_benchmark(syntax="cy", args=(100000,))
def numpy_typed_interp(int n):
    """Interpolate n query points and return sum."""
    rng = np.random.RandomState(42)
    cdef int num_knots = 1000
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xp_arr = (
        np.linspace(0.0, 1.0, num_knots).astype(
            np.float64
        )
    )
    cdef cnp.float64_t[::1] xp = xp_arr

    cdef cnp.ndarray[cnp.float64_t, ndim=1] fp_arr = (
        np.cumsum(
            rng.standard_normal(num_knots)
        ).astype(np.float64)
    )
    cdef cnp.float64_t[::1] fp = fp_arr

    cdef cnp.ndarray[cnp.float64_t, ndim=1] xq_arr = (
        rng.random(n).astype(np.float64)
    )
    cdef cnp.float64_t[::1] xq = xq_arr

    cdef cnp.float64_t total = 0.0
    cdef int i, idx
    cdef cnp.float64_t t, x_val

    with nogil:
        for i in range(n):
            x_val = xq[i]
            if x_val <= xp[0]:
                total += fp[0]
            elif x_val >= xp[num_knots - 1]:
                total += fp[num_knots - 1]
            else:
                idx = bisect(xp, num_knots, x_val)
                t = (
                    (x_val - xp[idx])
                    / (xp[idx + 1] - xp[idx])
                )
                total += (
                    fp[idx] + t * (fp[idx + 1] - fp[idx])
                )

    return total
