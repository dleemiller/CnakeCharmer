# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Layer norm f32 kernels — scalar and AVX2 implementations.

Pure compute, no allocation. Two-pass: compute mean and variance,
then normalize: out[i] = (inp[i] - mean) / sqrt(var + epsilon).
"""

from libc.math cimport sqrtf

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_sub_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_setzero_ps() noexcept
    # Horizontal sum intrinsics (XNNPACK hsum pattern)
    __m128 _mm256_castps256_ps128(__m256 a) noexcept
    __m128 _mm256_extractf128_ps(__m256 a, int imm) noexcept
    __m128 _mm_add_ps(__m128 a, __m128 b) noexcept
    __m128 _mm_add_ss(__m128 a, __m128 b) noexcept
    __m128 _mm_movehl_ps(__m128 a, __m128 b) noexcept
    __m128 _mm_movehdup_ps(__m128 a) noexcept
    float _mm_cvtss_f32(__m128 a) noexcept


cdef inline float _hsum_avx(__m256 v) noexcept nogil:
    """Horizontal sum — XNNPACK pattern, register-only."""
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 s = _mm_add_ps(lo, hi)
    s = _mm_add_ps(s, _mm_movehl_ps(s, s))
    s = _mm_add_ss(s, _mm_movehdup_ps(s))
    return _mm_cvtss_f32(s)


cdef void layer_norm_f32(const float *inp, float *out,
                         int n, float epsilon) noexcept nogil:
    """Scalar layer norm: two-pass mean+var, then normalize."""
    cdef int i
    cdef float mean = 0.0
    cdef float var = 0.0
    cdef float diff, inv_std

    # Pass 1: mean
    for i in range(n):
        mean += inp[i]
    mean /= n

    # Pass 2: variance
    for i in range(n):
        diff = inp[i] - mean
        var += diff * diff
    var /= n

    # Normalize
    inv_std = 1.0 / sqrtf(var + epsilon)
    for i in range(n):
        out[i] = (inp[i] - mean) * inv_std


cdef void layer_norm_f32_avx(const float *inp, float *out,
                             int n, float epsilon) noexcept nogil:
    """AVX2 layer norm: 4-accumulator rsum (XNNPACK pattern) + FMA variance."""
    cdef int i
    cdef int end32 = (n // 32) * 32
    cdef int end8 = (n // 8) * 8
    cdef __m256 vs0, vs1, vs2, vs3, vmean, vd, vinv
    cdef float mean, var_sum, inv_std

    # Pass 1: mean — 4 accumulators for ILP
    vs0 = _mm256_setzero_ps()
    vs1 = _mm256_setzero_ps()
    vs2 = _mm256_setzero_ps()
    vs3 = _mm256_setzero_ps()
    for i in range(0, end32, 32):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&inp[i]))
        vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(&inp[i + 8]))
        vs2 = _mm256_add_ps(vs2, _mm256_loadu_ps(&inp[i + 16]))
        vs3 = _mm256_add_ps(vs3, _mm256_loadu_ps(&inp[i + 24]))
    for i in range(end32, end8, 8):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&inp[i]))
    vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
    mean = _hsum_avx(vs0)
    for i in range(end8, n):
        mean += inp[i]
    mean /= n

    # Pass 2: variance — 4 accumulators with FMA
    vmean = _mm256_set1_ps(mean)
    vs0 = _mm256_setzero_ps()
    vs1 = _mm256_setzero_ps()
    vs2 = _mm256_setzero_ps()
    vs3 = _mm256_setzero_ps()
    for i in range(0, end32, 32):
        vd = _mm256_sub_ps(_mm256_loadu_ps(&inp[i]), vmean)
        vs0 = _mm256_fmadd_ps(vd, vd, vs0)
        vd = _mm256_sub_ps(_mm256_loadu_ps(&inp[i + 8]), vmean)
        vs1 = _mm256_fmadd_ps(vd, vd, vs1)
        vd = _mm256_sub_ps(_mm256_loadu_ps(&inp[i + 16]), vmean)
        vs2 = _mm256_fmadd_ps(vd, vd, vs2)
        vd = _mm256_sub_ps(_mm256_loadu_ps(&inp[i + 24]), vmean)
        vs3 = _mm256_fmadd_ps(vd, vd, vs3)
    for i in range(end32, end8, 8):
        vd = _mm256_sub_ps(_mm256_loadu_ps(&inp[i]), vmean)
        vs0 = _mm256_fmadd_ps(vd, vd, vs0)
    vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
    var_sum = _hsum_avx(vs0)
    for i in range(end8, n):
        var_sum += (inp[i] - mean) * (inp[i] - mean)

    # Pass 3: normalize
    inv_std = 1.0 / sqrtf(var_sum / n + epsilon)
    vinv = _mm256_set1_ps(inv_std)
    for i in range(0, end8, 8):
        vd = _mm256_sub_ps(_mm256_loadu_ps(&inp[i]), vmean)
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(vd, vinv))
    for i in range(end8, n):
        out[i] = (inp[i] - mean) * inv_std
