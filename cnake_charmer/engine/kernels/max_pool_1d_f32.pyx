# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Max pool 1D f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Max pooling with configurable kernel_size and stride.
Output length = (n - kernel_size) / stride + 1.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_max_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_set1_ps(float a) noexcept

cdef extern from "float.h" nogil:
    float FLT_MIN


cdef void max_pool_1d_f32(const float *inp, float *out,
                          int n, int kernel_size, int stride) noexcept nogil:
    """Scalar 1D max pooling."""
    cdef int out_len = (n - kernel_size) // stride + 1
    cdef int i, k
    cdef float mx

    for i in range(out_len):
        mx = inp[i * stride]
        for k in range(1, kernel_size):
            if inp[i * stride + k] > mx:
                mx = inp[i * stride + k]
        out[i] = mx


cdef void max_pool_1d_f32_avx(const float *inp, float *out,
                              int n, int kernel_size, int stride) noexcept nogil:
    """AVX2 1D max pool: _mm256_max_ps across kernel window.

    When stride == 1, we can vectorize 8 output positions at once.
    For arbitrary stride, falls back to scalar per output position
    with AVX max across the kernel window when kernel_size >= 8.
    """
    cdef int out_len = (n - kernel_size) // stride + 1
    cdef int i, k
    cdef float mx
    cdef __m256 vmax, vtmp
    cdef int end8

    if stride == 1:
        # stride-1 fast path: 8 outputs at a time
        end8 = (out_len // 8) * 8
        for i in range(0, end8, 8):
            # Start with first kernel position
            vmax = _mm256_loadu_ps(&inp[i])
            for k in range(1, kernel_size):
                vtmp = _mm256_loadu_ps(&inp[i + k])
                vmax = _mm256_max_ps(vmax, vtmp)
            _mm256_storeu_ps(&out[i], vmax)

        # Scalar remainder
        for i in range(end8, out_len):
            mx = inp[i]
            for k in range(1, kernel_size):
                if inp[i + k] > mx:
                    mx = inp[i + k]
            out[i] = mx
    else:
        # General stride: scalar per output element
        for i in range(out_len):
            mx = inp[i * stride]
            for k in range(1, kernel_size):
                if inp[i * stride + k] > mx:
                    mx = inp[i * stride + k]
            out[i] = mx
