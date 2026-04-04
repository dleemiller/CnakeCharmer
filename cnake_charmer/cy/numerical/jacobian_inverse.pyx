# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Batch inversion of 2×2 Jacobian matrices (Cython-optimized).

Keywords: numerical, jacobian, matrix inversion, fem, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def jacobian_inverse(int n):
    """Batch-invert n*4 Jacobian 2x2 matrices using C arrays and nogil."""
    cdef int i, j, k, m, pos
    cdef double a, b, c, d, f, sum_diag
    m = 4
    cdef int size = n * m * 4

    cdef double *J = <double *>malloc(size * sizeof(double))
    cdef double *invJ = <double *>malloc(size * sizeof(double))

    if not J or not invJ:
        if J: free(J)
        if invJ: free(invJ)
        raise MemoryError("Failed to allocate arrays")

    with nogil:
        # Fill J
        for i in range(n):
            for j in range(m):
                pos = m * 4 * i + 4 * j
                for k in range(4):
                    J[pos + k] = (i * 1000 + j * 100 + k + 1) * 0.01

        # Invert each 2x2 Jacobian
        sum_diag = 0.0
        for i in range(n):
            for j in range(m):
                pos = m * 4 * i + 4 * j
                a = J[pos + 0]
                b = J[pos + 1]
                c = J[pos + 2]
                d = J[pos + 3]
                f = 1.0 / (a * d - b * c)
                invJ[pos + 0] = f * d
                invJ[pos + 1] = -f * b
                invJ[pos + 2] = -f * c
                invJ[pos + 3] = f * a
                sum_diag += invJ[pos + 0]

    cdef int mid_pos = m * 4 * (n // 2) + 0
    cdef double invJ_mid = invJ[mid_pos]
    cdef long long r0 = (<long long>(sum_diag * 1e6)) % (10 ** 9)
    cdef long long r1 = <long long>(invJ_mid * 1e9)

    free(J)
    free(invJ)
    # Python % always returns non-negative; C % can be negative for negative values
    if r0 < 0:
        r0 += 10 ** 9
    return r0, r1
