# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""GEMM — full C = A * B with f32 C arrays (basic Cython, no SIMD).

Same algorithm as SIMD version but scalar operations.
This is the baseline the XNNPACK-style 4x8 FMA kernel should beat.

Keywords: gemm, matrix multiply, f32, linear algebra, neural network, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def gemm(int n):
    """Full C = A * B on f32 flat C arrays, return trace(C)."""
    cdef float *A = <float *>malloc(n * n * sizeof(float))
    cdef float *B = <float *>malloc(n * n * sizeof(float))
    cdef float *C = <float *>malloc(n * n * sizeof(float))
    if not A or not B or not C:
        if A: free(A)
        if B: free(B)
        if C: free(C)
        raise MemoryError()

    cdef int i, j, k
    cdef float s
    cdef double trace

    # Initialize
    for i in range(n):
        for j in range(n):
            A[i * n + j] = <float>((i + j) % 100) / 10.0
            B[i * n + j] = <float>((i - j + n) % 100) / 10.0

    memset(C, 0, n * n * sizeof(float))

    # Scalar triple loop (i-k-j for cache friendliness)
    for i in range(n):
        for k in range(n):
            s = A[i * n + k]
            for j in range(n):
                C[i * n + j] += s * B[k * n + j]

    trace = 0.0
    for i in range(n):
        trace += C[i * n + i]

    free(A)
    free(B)
    free(C)
    return trace
