# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Single-pass statistics computation (Cython-optimized).

Keywords: statistics, single_pass, mean, stddev, min_max, efficient, cython
"""

from cnake_charmer.benchmarks import cython_benchmark
from libc.math cimport sqrt


@cython_benchmark(syntax="cy", args=(500000,))
def stats_single_pass(int n):
    """Compute statistics on a generated list of n floats in one pass."""
    cdef list data = []
    cdef double x = 1.0
    cdef int i
    cdef double val

    for i in range(n):
        x = (x * 1103515245 + 12345) % 2147483648.0
        data.append(x / 2147483648.0 * 100.0)

    cdef double min_val = <double>data[0]
    cdef double max_val = <double>data[0]
    cdef double total = 0.0
    cdef double sum_sq = 0.0

    for i in range(n):
        val = <double>data[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
        total += val
        sum_sq += val * val

    cdef double mean = total / n
    cdef double pstdev = sqrt((sum_sq / n) - (mean * mean))

    return {
        "len": n,
        "min": min_val,
        "max": max_val,
        "sum": total,
        "mean": mean,
        "pstdev": pstdev,
    }
