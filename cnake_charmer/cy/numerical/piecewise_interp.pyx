# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Piecewise linear interpolation from a table of (x, y) breakpoints (Cython).

Given n query points, use binary search to find the bracketing interval and
interpolate.  Returns (sum, max) of all interpolated values.

Keywords: numerical, interpolation, piecewise linear, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def piecewise_interp(int n):
    """Evaluate piecewise linear interpolation at n query points."""
    cdef int i, lo, hi, mid
    cdef int num_bp = 50
    cdef double t, val, x1, x2, y1, y2
    cdef double total_sum = 0.0
    cdef double max_val = -1e300
    cdef double *x_bp = <double *>malloc(num_bp * sizeof(double))
    cdef double *y_bp = <double *>malloc(num_bp * sizeof(double))

    if not x_bp or not y_bp:
        raise MemoryError()

    # Build breakpoint table
    for i in range(num_bp):
        x_bp[i] = i * 2.0
        y_bp[i] = ((i * i * 7 + 3) % 100) / 10.0

    with nogil:
        for i in range(n):
            t = (i * 97.0 / n) % 98.0

            # Clamp below
            if t <= x_bp[0]:
                val = y_bp[0]
            # Clamp above
            elif t >= x_bp[num_bp - 1]:
                val = y_bp[num_bp - 1]
            else:
                # Binary search for bracketing interval
                lo = 0
                hi = num_bp - 1
                while hi - lo > 1:
                    mid = (lo + hi) >> 1
                    if x_bp[mid] <= t:
                        lo = mid
                    else:
                        hi = mid
                x1 = x_bp[lo]
                x2 = x_bp[lo + 1]
                y1 = y_bp[lo]
                y2 = y_bp[lo + 1]
                val = y1 + (y2 - y1) / (x2 - x1) * (t - x1)

            total_sum += val
            if val > max_val:
                max_val = val

    free(x_bp)
    free(y_bp)
    return (total_sum, max_val)
