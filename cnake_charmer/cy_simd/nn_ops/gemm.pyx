# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""General matrix multiply (GEMM) with AVX2 FMA and cache tiling.

Computes C = A * B using the XNNPACK-style pattern:
- Tiled loop for cache efficiency (TILE=32)
- AVX2 FMA: broadcast A[i][k], multiply with 4 packed B values, accumulate

Keywords: matrix multiply, gemm, linear algebra, simd, avx2, fma, tiling, cython
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h":
    ctypedef double __m256d
    __m256d _mm256_loadu_pd(const double *mem)
    __m256d _mm256_broadcast_sd(const double *mem)
    __m256d _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c)
    __m256d _mm256_setzero_pd()
    void _mm256_storeu_pd(double *mem, __m256d a)


@cython_benchmark(syntax="cy_simd", args=(300,))
def gemm(int n):
    """Compute C = A * B for n×n matrices, return trace(C).

    Uses AVX2 FMA with 32×32 cache tiling.
    """
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    cdef double *B = <double *>malloc(n * n * sizeof(double))
    cdef double *C = <double *>malloc(n * n * sizeof(double))
    if not A or not B or not C:
        if A: free(A)
        if B: free(B)
        if C: free(C)
        raise MemoryError()

    cdef int i, j, k
    cdef int ii, jj, kk
    cdef int TILE = 32
    cdef int i_end, j_end, k_end

    # Initialize matrices
    for i in range(n):
        for j in range(n):
            A[i * n + j] = ((i + j) % 100) / 10.0
            B[i * n + j] = ((i - j + n) % 100) / 10.0

    memset(C, 0, n * n * sizeof(double))

    # Tiled GEMM with AVX2 FMA
    cdef __m256d a_broadcast, b_vec, c_vec
    cdef int simd_end

    for ii in range(0, n, TILE):
        i_end = ii + TILE if ii + TILE < n else n
        for kk in range(0, n, TILE):
            k_end = kk + TILE if kk + TILE < n else n
            for jj in range(0, n, TILE):
                j_end = jj + TILE if jj + TILE < n else n
                simd_end = jj + ((j_end - jj) // 4) * 4

                for i in range(ii, i_end):
                    for k in range(kk, k_end):
                        # XNNPACK pattern: broadcast A[i][k], FMA with B[k][j..j+3]
                        a_broadcast = _mm256_broadcast_sd(&A[i * n + k])
                        for j in range(jj, simd_end, 4):
                            b_vec = _mm256_loadu_pd(&B[k * n + j])
                            c_vec = _mm256_loadu_pd(&C[i * n + j])
                            c_vec = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec)
                            _mm256_storeu_pd(&C[i * n + j], c_vec)
                        # Scalar remainder
                        for j in range(simd_end, j_end):
                            C[i * n + j] += A[i * n + k] * B[k * n + j]

    # Extract trace
    cdef double trace = 0.0
    for i in range(n):
        trace += C[i * n + i]

    free(A)
    free(B)
    free(C)
    return trace
