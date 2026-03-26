# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""GEMM f32 kernels — scalar and XNNPACK-style 4x8 AVX2 FMA.

Pure compute, no allocation. Caller provides pre-allocated A, B, C.
C must be pre-zeroed (or accumulated into).
"""

cdef extern from "string.h" nogil:
    void *memset(void *s, int c, size_t n) noexcept

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_broadcast_ss(const float *mem) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void gemm_f32(const float *A, const float *B, float *C,
                   int M, int N, int K) noexcept nogil:
    """Scalar GEMM: C += A * B. i-k-j loop order for cache friendliness."""
    cdef int i, j, k
    cdef float a_ik

    memset(C, 0, M * N * sizeof(float))

    for i in range(M):
        for k in range(K):
            a_ik = A[i * K + k]
            for j in range(N):
                C[i * N + j] += a_ik * B[k * N + j]


cdef void gemm_f32_avx(const float *A, const float *B, float *C,
                       int M, int N, int K) noexcept nogil:
    """XNNPACK-style 4x8 register-blocked GEMM with AVX2 FMA.

    Processes 4 rows x 8 columns of C per tile. Inner loop broadcasts
    A[row][k] and FMAs with 8-packed B[k][col..col+7].
    """
    cdef int i, j, k
    cdef int nr = (N // 8) * 8
    cdef __m256 vacc0, vacc1, vacc2, vacc3
    cdef __m256 va0, va1, va2, va3, vb

    memset(C, 0, M * N * sizeof(float))

    # 4x8 tiled main loop
    for i in range(0, M - 3, 4):
        for j in range(0, nr, 8):
            vacc0 = _mm256_setzero_ps()
            vacc1 = _mm256_setzero_ps()
            vacc2 = _mm256_setzero_ps()
            vacc3 = _mm256_setzero_ps()

            for k in range(K):
                va0 = _mm256_broadcast_ss(&A[(i + 0) * K + k])
                va1 = _mm256_broadcast_ss(&A[(i + 1) * K + k])
                va2 = _mm256_broadcast_ss(&A[(i + 2) * K + k])
                va3 = _mm256_broadcast_ss(&A[(i + 3) * K + k])
                vb = _mm256_loadu_ps(&B[k * N + j])

                vacc0 = _mm256_fmadd_ps(va0, vb, vacc0)
                vacc1 = _mm256_fmadd_ps(va1, vb, vacc1)
                vacc2 = _mm256_fmadd_ps(va2, vb, vacc2)
                vacc3 = _mm256_fmadd_ps(va3, vb, vacc3)

            # Store tile (add to C in case of accumulation)
            _mm256_storeu_ps(&C[(i + 0) * N + j],
                             _mm256_add_ps(_mm256_loadu_ps(&C[(i + 0) * N + j]), vacc0))
            _mm256_storeu_ps(&C[(i + 1) * N + j],
                             _mm256_add_ps(_mm256_loadu_ps(&C[(i + 1) * N + j]), vacc1))
            _mm256_storeu_ps(&C[(i + 2) * N + j],
                             _mm256_add_ps(_mm256_loadu_ps(&C[(i + 2) * N + j]), vacc2))
            _mm256_storeu_ps(&C[(i + 3) * N + j],
                             _mm256_add_ps(_mm256_loadu_ps(&C[(i + 3) * N + j]), vacc3))

        # Remainder columns (scalar)
        for j in range(nr, N):
            for k in range(K):
                C[(i + 0) * N + j] += A[(i + 0) * K + k] * B[k * N + j]
                C[(i + 1) * N + j] += A[(i + 1) * K + k] * B[k * N + j]
                C[(i + 2) * N + j] += A[(i + 2) * K + k] * B[k * N + j]
                C[(i + 3) * N + j] += A[(i + 3) * K + k] * B[k * N + j]

    # Remainder rows (scalar)
    for i in range((M // 4) * 4, M):
        for k in range(K):
            for j in range(N):
                C[i * N + j] += A[i * K + k] * B[k * N + j]
