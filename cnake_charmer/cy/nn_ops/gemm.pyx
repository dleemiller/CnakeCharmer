# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""General matrix multiply (GEMM) with trace computation.

Keywords: matrix multiply, gemm, linear algebra, neural network, blas, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def gemm(int n):
    """Compute C = A * B for n x n matrices and return trace(C)."""
    cdef int i, k
    cdef double trace = 0.0
    cdef double s
    cdef double a_ik, b_ki

    # Only need trace: C[i][i] = sum_k A[i][k] * B[k][i]
    for i in range(n):
        s = 0.0
        for k in range(n):
            a_ik = ((i + k) % 100) / 10.0
            b_ki = ((k - i + n) % 100) / 10.0
            s += a_ik * b_ki
        trace += s
    return trace
