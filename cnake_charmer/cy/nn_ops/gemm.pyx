# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""General matrix multiply trace (basic Cython, no SIMD).

Computes trace(A*B) directly without materializing the full product.
trace(C) = sum_i sum_k A[i][k] * B[k][i]

Keywords: matrix multiply, gemm, linear algebra, neural network, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def gemm(int n):
    """Compute trace(A*B) for n×n matrices using typed inner loop."""
    cdef int i, k
    cdef double trace = 0.0
    cdef double s, a_ik, b_ki

    for i in range(n):
        s = 0.0
        for k in range(n):
            a_ik = ((i + k) % 100) / 10.0
            b_ki = ((k - i + n) % 100) / 10.0
            s += a_ik * b_ki
        trace += s

    return trace
