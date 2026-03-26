# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""GEMM — XNNPACK-style 4x8 register-blocked AVX2 FMA microkernel.

Directly follows the f32-gemm-4x8-minmax-avx-broadcast pattern:
- 4 rows of accumulators (vacc0..vacc3), each 8-wide (__m256)
- Inner loop: broadcast A[row][k], FMA with packed B[k][col..col+7]
- Outer loop tiles over columns in steps of 8

This is how production inference engines (XNNPACK, oneDNN, BLAS)
implement the innermost GEMM kernel.

Keywords: gemm, matrix multiply, simd, avx2, fma, xnnpack, microkernel, cython
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_broadcast_ss(const float *mem)
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_setzero_ps()


@cython_benchmark(syntax="cy_simd", args=(200,))
def gemm(int n):
    """Full C=A*B with XNNPACK-style 4x8 register-blocked AVX2 FMA."""
    cdef float *A = <float *>malloc(n * n * sizeof(float))
    cdef float *B = <float *>malloc(n * n * sizeof(float))
    cdef float *C = <float *>malloc(n * n * sizeof(float))
    if not A or not B or not C:
        if A: free(A)
        if B: free(B)
        if C: free(C)
        raise MemoryError()

    cdef int i, j, k
    cdef double trace

    # Initialize
    for i in range(n):
        for j in range(n):
            A[i * n + j] = <float>((i + j) % 100) / 10.0
            B[i * n + j] = <float>((i - j + n) % 100) / 10.0

    memset(C, 0, n * n * sizeof(float))

    # XNNPACK 4x8 microkernel pattern
    # Process 4 rows × 8 columns of C at a time
    cdef __m256 vacc0, vacc1, vacc2, vacc3
    cdef __m256 va0, va1, va2, va3, vb
    cdef int i0, col_end
    cdef int nr = (n // 8) * 8  # columns rounded down to 8

    for i0 in range(0, n - 3, 4):
        for j in range(0, nr, 8):
            # Zero accumulators (4 rows × 8 cols)
            vacc0 = _mm256_setzero_ps()
            vacc1 = _mm256_setzero_ps()
            vacc2 = _mm256_setzero_ps()
            vacc3 = _mm256_setzero_ps()

            # Inner product loop over k
            for k in range(n):
                # Broadcast A[row][k] for each of 4 rows
                va0 = _mm256_broadcast_ss(&A[(i0 + 0) * n + k])
                va1 = _mm256_broadcast_ss(&A[(i0 + 1) * n + k])
                va2 = _mm256_broadcast_ss(&A[(i0 + 2) * n + k])
                va3 = _mm256_broadcast_ss(&A[(i0 + 3) * n + k])

                # Load 8 packed B values
                vb = _mm256_loadu_ps(&B[k * n + j])

                # FMA: acc += a * b
                vacc0 = _mm256_fmadd_ps(va0, vb, vacc0)
                vacc1 = _mm256_fmadd_ps(va1, vb, vacc1)
                vacc2 = _mm256_fmadd_ps(va2, vb, vacc2)
                vacc3 = _mm256_fmadd_ps(va3, vb, vacc3)

            # Store 4×8 tile
            _mm256_storeu_ps(&C[(i0 + 0) * n + j], _mm256_add_ps(
                _mm256_loadu_ps(&C[(i0 + 0) * n + j]), vacc0))
            _mm256_storeu_ps(&C[(i0 + 1) * n + j], _mm256_add_ps(
                _mm256_loadu_ps(&C[(i0 + 1) * n + j]), vacc1))
            _mm256_storeu_ps(&C[(i0 + 2) * n + j], _mm256_add_ps(
                _mm256_loadu_ps(&C[(i0 + 2) * n + j]), vacc2))
            _mm256_storeu_ps(&C[(i0 + 3) * n + j], _mm256_add_ps(
                _mm256_loadu_ps(&C[(i0 + 3) * n + j]), vacc3))

        # Scalar remainder columns
        for j in range(nr, n):
            for k in range(n):
                C[(i0 + 0) * n + j] += A[(i0 + 0) * n + k] * B[k * n + j]
                C[(i0 + 1) * n + j] += A[(i0 + 1) * n + k] * B[k * n + j]
                C[(i0 + 2) * n + j] += A[(i0 + 2) * n + k] * B[k * n + j]
                C[(i0 + 3) * n + j] += A[(i0 + 3) * n + k] * B[k * n + j]

    # Remainder rows (< 4)
    for i in range((n // 4) * 4, n):
        for k in range(n):
            for j in range(n):
                C[i * n + j] += A[i * n + k] * B[k * n + j]

    # Extract trace
    trace = 0.0
    for i in range(n):
        trace += C[i * n + i]

    free(A)
    free(B)
    free(C)
    return trace
