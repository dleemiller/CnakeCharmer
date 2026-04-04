# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Z-score normalize a deterministic dataset (Cython-optimized).

Keywords: statistics, z-score, normalization, standardize, mean, stddev, cython, benchmark
"""

from libc.math cimport cos, sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def z_score_normalize(int n):
    """Z-score normalize a dataset and return summary statistics."""
    cdef int i
    cdef double total = 0.0, mean, val, d, var_sum = 0.0, stddev
    cdef double norm_sum = 0.0, norm_max = -1e308, norm_min = 1e308, z

    # Pass 1: compute mean
    for i in range(n):
        total += cos(i * 0.007) * 30.0 + (i * 23 + 3) % 151
    mean = total / n

    # Pass 2: compute variance
    for i in range(n):
        val = cos(i * 0.007) * 30.0 + (i * 23 + 3) % 151
        d = val - mean
        var_sum += d * d
    stddev = sqrt(var_sum / n)

    if stddev == 0.0:
        return (0.0, 0.0, 0.0)

    # Pass 3: normalize and compute summary
    for i in range(n):
        val = cos(i * 0.007) * 30.0 + (i * 23 + 3) % 151
        z = (val - mean) / stddev
        norm_sum += z
        if z > norm_max:
            norm_max = z
        if z < norm_min:
            norm_min = z

    return (norm_sum, norm_max, norm_min)
