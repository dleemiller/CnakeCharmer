# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Conv1D f32 kernels — scalar and AVX2 FMA implementations.

Pure compute, no allocation. output[i] = sum(input[i+k] * kernel[k]).
Output length = n - kernel_size + 1.
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_broadcast_ss(const float *mem) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void conv1d_f32(const float *inp, const float *kernel,
                     float *out, int n, int kernel_size) noexcept nogil:
    """Scalar 1D convolution."""
    cdef int i, k
    cdef int out_len = n - kernel_size + 1
    cdef float acc

    for i in range(out_len):
        acc = 0.0
        for k in range(kernel_size):
            acc += inp[i + k] * kernel[k]
        out[i] = acc


cdef void conv1d_f32_avx(const float *inp, const float *kernel,
                         float *out, int n, int kernel_size) noexcept nogil:
    """AVX2 1D convolution: process 8 output elements at a time with FMA.

    For each kernel tap, broadcast kernel[k] and FMA with 8 input elements.
    """
    cdef int i, k
    cdef int out_len = n - kernel_size + 1
    cdef int end8 = (out_len // 8) * 8
    cdef __m256 vacc, vk, vinp

    # Vectorized: 8 outputs at a time
    for i in range(0, end8, 8):
        vacc = _mm256_setzero_ps()
        for k in range(kernel_size):
            vk = _mm256_broadcast_ss(&kernel[k])
            vinp = _mm256_loadu_ps(&inp[i + k])
            vacc = _mm256_fmadd_ps(vk, vinp, vacc)
        _mm256_storeu_ps(&out[i], vacc)

    # Scalar remainder
    cdef float acc
    for i in range(end8, out_len):
        acc = 0.0
        for k in range(kernel_size):
            acc += inp[i + k] * kernel[k]
        out[i] = acc
