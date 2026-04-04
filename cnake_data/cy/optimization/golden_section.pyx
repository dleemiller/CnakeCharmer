# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Golden section search for minimizing n different unimodal functions (Cython-optimized).

Each function is a shifted quadratic with deterministic coefficients.
Returns summary of minima found.

Keywords: optimization, golden section, search, minimization, unimodal, cython, benchmark
"""

from libc.math cimport sin, sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def golden_section(int n):
    """Find minima of n unimodal functions using golden section search."""
    cdef double gr = (sqrt(5.0) + 1.0) / 2.0
    cdef int max_iter = 200
    cdef double tol = 1e-12
    cdef double sum_minima = 0.0, min_of_first = 0.0, min_of_last = 0.0
    cdef double a, b, c, lo, hi, d, x1, x2, f1, f2, x_min, f_min
    cdef int idx, it

    for idx in range(n):
        # Deterministic coefficients
        a = 1.0 + (idx % 10) * 0.5
        b = 0.3 * sin(idx * 0.1)
        c = -5.0 + 10.0 * ((idx * 7 + 3) % n) / n if n > 0 else 0.0

        # Search interval around c
        lo = c - 10.0
        hi = c + 10.0

        for it in range(max_iter):
            if hi - lo < tol:
                break
            d = (hi - lo) / gr
            x1 = hi - d
            x2 = lo + d

            f1 = a * (x1 - c) * (x1 - c) + b * sin(x1 - c)
            f2 = a * (x2 - c) * (x2 - c) + b * sin(x2 - c)

            if f1 < f2:
                hi = x2
            else:
                lo = x1

        x_min = (lo + hi) / 2.0
        f_min = a * (x_min - c) * (x_min - c) + b * sin(x_min - c)
        sum_minima += f_min

        if idx == 0:
            min_of_first = f_min
        if idx == n - 1:
            min_of_last = f_min

    return (sum_minima, min_of_first, min_of_last)
