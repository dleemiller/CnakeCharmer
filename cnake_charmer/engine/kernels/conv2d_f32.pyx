# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Conv2D f32 kernels — scalar and AVX2 FMA implementations.

Pure compute, no allocation. 2D convolution on flat row-major arrays.
Output dimensions: (in_h - kh + 1) x (in_w - kw + 1).
"""

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_broadcast_ss(const float *mem) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef void conv2d_f32(const float *inp, const float *kernel, float *out,
                     int in_h, int in_w, int kh, int kw) noexcept nogil:
    """Scalar 2D convolution on flat arrays."""
    cdef int oh = in_h - kh + 1
    cdef int ow = in_w - kw + 1
    cdef int i, j, ki, kj
    cdef float acc

    for i in range(oh):
        for j in range(ow):
            acc = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acc += inp[(i + ki) * in_w + (j + kj)] * kernel[ki * kw + kj]
            out[i * ow + j] = acc


cdef void conv2d_f32_avx(const float *inp, const float *kernel, float *out,
                         int in_h, int in_w, int kh, int kw) noexcept nogil:
    """AVX2 2D convolution: FMA in inner kernel loop, 8 output cols at a time."""
    cdef int oh = in_h - kh + 1
    cdef int ow = in_w - kw + 1
    cdef int end8 = (ow // 8) * 8
    cdef int i, j, ki, kj
    cdef __m256 vacc, vk, vinp
    cdef float acc

    for i in range(oh):
        # Vectorized columns
        for j in range(0, end8, 8):
            vacc = _mm256_setzero_ps()
            for ki in range(kh):
                for kj in range(kw):
                    vk = _mm256_broadcast_ss(&kernel[ki * kw + kj])
                    vinp = _mm256_loadu_ps(&inp[(i + ki) * in_w + (j + kj)])
                    vacc = _mm256_fmadd_ps(vk, vinp, vacc)
            _mm256_storeu_ps(&out[i * ow + j], vacc)

        # Scalar remainder columns
        for j in range(end8, ow):
            acc = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acc += inp[(i + ki) * in_w + (j + kj)] * kernel[ki * kw + kj]
            out[i * ow + j] = acc
