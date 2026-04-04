# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Evaluate renormalized associated Legendre polynomials at many x values and sum (Cython).

Keywords: legendre, polynomial, math, recurrence, cython, benchmark
"""

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark

cdef double PI = 3.14159265358979323846


cdef double _assoc_legendre(long l, long m, double x) noexcept nogil:
    cdef long i, ll
    cdef double fact, oldfact, pll, pmm, pmmp1, omx2

    pmm = 1.0
    if m > 0:
        omx2 = (1.0 - x) * (1.0 + x)
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= omx2 * fact / (fact + 1.0)
            fact += 2.0
    pmm = sqrt((2 * m + 1) * pmm / (4.0 * PI))
    if m & 1:
        pmm = -pmm
    if l == m:
        return pmm

    pmmp1 = x * sqrt(2.0 * m + 3.0) * pmm
    if l == (m + 1):
        return pmmp1

    oldfact = sqrt(2.0 * m + 3.0)
    pll = 0.0
    for ll in range(m + 2, l + 1):
        fact = sqrt((4.0 * ll * ll - 1.0) / (ll * ll - m * m))
        pll = (x * pmmp1 - pmm / oldfact) * fact
        oldfact = fact
        pmm = pmmp1
        pmmp1 = pll
    return pll


@cython_benchmark(syntax="cy", args=(50000,))
def assoc_legendre_sum(int n):
    """Evaluate renormalized associated Legendre P_10^3(x) at n points and sum."""
    cdef long l = 10
    cdef long m = 3
    cdef double total = 0.0
    cdef double mid_val = 0.0
    cdef int mid_idx = n // 2
    cdef int i
    cdef double x, val

    with nogil:
        for i in range(n):
            if n > 1:
                x = -0.99 + 1.98 * i / (n - 1)
            else:
                x = 0.0
            val = _assoc_legendre(l, m, x)
            total += val
            if i == mid_idx:
                mid_val = val

    return (total, mid_val)
