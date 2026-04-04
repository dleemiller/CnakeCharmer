# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate generalized Laguerre polynomials iteratively over many x values (Cython-optimized).

Keywords: laguerre, polynomial, numerical, iterative, orthogonal, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def laguerre_polynomial(int n):
    """Evaluate L_k^alpha(x) for k=15, alpha=0.5 at n evenly-spaced x in [0, 20]."""
    cdef int k = 15
    cdef double alpha = 0.5
    cdef double total = 0.0
    cdef double mid_val = 0.0
    cdef int mid_idx = n // 2
    cdef int idx, i
    cdef double x, minus_2, minus_1, a, b, current, val
    cdef double inv_nm1

    if n > 1:
        inv_nm1 = 20.0 / (n - 1)
    else:
        inv_nm1 = 0.0

    with nogil:
        for idx in range(n):
            x = inv_nm1 * idx

            # Evaluate L_k^alpha(x) via three-term recurrence
            minus_2 = 1.0
            minus_1 = -x + alpha + 1.0
            for i in range(2, k + 1):
                a = (2.0 * i - 1.0 + alpha - x) / i
                b = (i + alpha - 1.0) / i
                current = a * minus_1 - b * minus_2
                minus_2 = minus_1
                minus_1 = current

            val = minus_1
            total = total + val
            if idx == mid_idx:
                mid_val = val

    return (total, mid_val)
