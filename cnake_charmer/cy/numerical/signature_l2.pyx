# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Weighted L2 distance between deterministic signatures (Cython).

Sourced from SFT DuckDB blob: 1949531910e5e150b1398a5d9580c37bb6e917c9
Keywords: signature, euclidean distance, l2, weighted norm, numerical, cython
"""

from libc.math cimport sqrt

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000, 11, 0.35))
def signature_l2(int n, int stride, double bias):
    cdef int i
    cdef double a, b, diff, ad
    cdef double sse = 0.0
    cdef double abs_sum = 0.0
    cdef double max_abs = 0.0
    cdef double dist, mean_abs

    for i in range(n):
        a = ((i * stride + 3) % 1000) * 0.01 + bias
        b = ((i * (stride + 4) + 17) % 1000) * 0.01 - bias * 0.5
        diff = a - b
        ad = diff if diff >= 0.0 else -diff
        sse += diff * diff
        abs_sum += ad
        if ad > max_abs:
            max_abs = ad

    dist = sqrt(sse)
    mean_abs = abs_sum / n if n > 0 else 0.0
    return (round(dist, 10), round(mean_abs, 10), round(max_abs, 10))
