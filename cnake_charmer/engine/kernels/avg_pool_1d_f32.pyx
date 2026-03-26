# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Average pooling 1D f32 kernels -- scalar and AVX2 implementations.

Average pool with configurable kernel and stride.
Pure compute, no allocation.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void avg_pool_1d_f32(const float *inp, float *out, int n, int kernel, int stride) noexcept nogil:
    """Scalar average pooling 1D."""
    cdef int out_len = n // stride
    cdef int i, k, base
    cdef float s, inv_k
    inv_k = 1.0 / kernel
    for i in range(out_len):
        base = i * stride
        s = 0.0
        for k in range(kernel):
            s += inp[base + k]
        out[i] = s * inv_k


cdef void avg_pool_1d_f32_avx(const float *inp, float *out, int n, int kernel, int stride) noexcept nogil:
    """AVX2 average pooling -- scalar fallback (stride pattern not vectorizable simply)."""
    avg_pool_1d_f32(inp, out, n, kernel, stride)


cdef double reduce_sum_f32(const float *data, int n) noexcept nogil:
    """Scalar sum reduction."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total


cdef double reduce_sum_f32_avx(const float *data, int n) noexcept nogil:
    """AVX2 sum reduction."""
    cdef __m256 acc = _mm256_setzero_ps()
    cdef int i
    cdef int end8 = (n // 8) * 8
    cdef float tmp[8]
    cdef double total = 0.0

    for i in range(0, end8, 8):
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&data[i]))

    _mm256_storeu_ps(tmp, acc)
    for i in range(8):
        total += tmp[i]
    for i in range(end8, n):
        total += data[i]

    return total
