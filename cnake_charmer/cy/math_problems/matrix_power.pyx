# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of traces of matrix powers (Cython-optimized).

Keywords: matrix, exponentiation, trace, linear algebra, modular arithmetic, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(100000,))
def matrix_power(int n):
    """Compute sum of traces of M^1 through M^n mod 10^9+7.

    M = [[1,1,0],[1,0,1],[0,1,1]]. Uses flat C arrays for 3x3 matrices.

    Args:
        n: Number of matrix powers to sum traces of.

    Returns:
        Sum of tr(M^1) + ... + tr(M^n), mod 10^9+7.
    """
    cdef long long cur[9]
    cdef long long new[9]
    cdef long long mat[9]
    cdef long long s, total, trace
    cdef int step, i, j, k

    # M = [[1,1,0],[1,0,1],[0,1,1]] stored row-major
    mat[0] = 1; mat[1] = 1; mat[2] = 0
    mat[3] = 1; mat[4] = 0; mat[5] = 1
    mat[6] = 0; mat[7] = 1; mat[8] = 1

    # cur = M^1
    for i in range(9):
        cur[i] = mat[i]

    total = 0
    for step in range(n):
        # Add trace
        trace = (cur[0] + cur[4] + cur[8]) % MOD
        total = (total + trace) % MOD

        # cur = cur * M
        for i in range(3):
            for j in range(3):
                s = 0
                for k in range(3):
                    s = s + cur[i * 3 + k] * mat[k * 3 + j]
                new[i * 3 + j] = s % MOD
        for i in range(9):
            cur[i] = new[i]

    return int(total)
