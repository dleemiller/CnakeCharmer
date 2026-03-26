# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Depthwise conv 1D f32 kernels -- scalar and AVX2 implementations.

Each channel convolved independently with its own kernel.
Pure compute, no allocation.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_broadcast_ss(const float *mem) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void depthwise_conv_f32(const float *inp, const float *kernel, float *out,
                             int channels, int spatial, int kernel_size) noexcept nogil:
    """Scalar depthwise 1D convolution."""
    cdef int c, s, k
    cdef int out_spatial = spatial - kernel_size + 1
    cdef int inp_offset, out_offset
    cdef float val

    for c in range(channels):
        inp_offset = c * spatial
        out_offset = c * out_spatial
        for s in range(out_spatial):
            val = 0.0
            for k in range(kernel_size):
                val += inp[inp_offset + s + k] * kernel[c * kernel_size + k]
            out[out_offset + s] = val


cdef void depthwise_conv_f32_avx(const float *inp, const float *kernel, float *out,
                                  int channels, int spatial, int kernel_size) noexcept nogil:
    """AVX2+FMA depthwise conv — broadcast kernel tap, 8 output positions at a time."""
    cdef int c, s, k
    cdef int out_spatial = spatial - kernel_size + 1
    cdef int inp_offset, kern_offset, out_offset
    cdef __m256 vacc, vk
    cdef float val

    for c in range(channels):
        inp_offset = c * spatial
        kern_offset = c * kernel_size
        out_offset = c * out_spatial
        s = 0
        while s + 8 <= out_spatial:
            vacc = _mm256_setzero_ps()
            for k in range(kernel_size):
                vk = _mm256_broadcast_ss(&kernel[kern_offset + k])
                vacc = _mm256_fmadd_ps(vk, _mm256_loadu_ps(&inp[inp_offset + s + k]), vacc)
            _mm256_storeu_ps(&out[out_offset + s], vacc)
            s += 8
        while s < out_spatial:
            val = 0.0
            for k in range(kernel_size):
                val += inp[inp_offset + s + k] * kernel[kern_offset + k]
            out[out_offset + s] = val
            s += 1


cdef double reduce_sum_f32(const float *data, int n) noexcept nogil:
    """Scalar sum reduction."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total
