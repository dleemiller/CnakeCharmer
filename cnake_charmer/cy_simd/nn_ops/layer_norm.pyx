# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Layer normalization — AVX2+FMA with 4-accumulator rsum pattern.

Keywords: layer norm, normalization, transformer, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrtf
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_loadu_ps(const float *mem)
    void _mm256_storeu_ps(float *mem, __m256 a)
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_sub_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
    __m256 _mm256_set1_ps(float a)
    __m256 _mm256_setzero_ps()
    __m128 _mm256_castps256_ps128(__m256 a)
    __m128 _mm256_extractf128_ps(__m256 a, int imm)
    __m128 _mm_add_ps(__m128 a, __m128 b)
    __m128 _mm_add_ss(__m128 a, __m128 b)
    __m128 _mm_movehl_ps(__m128 a, __m128 b)
    __m128 _mm_movehdup_ps(__m128 a)
    float _mm_cvtss_f32(__m128 a)


cdef inline float _hsum_avx(__m256 v) noexcept nogil:
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 s = _mm_add_ps(lo, hi)
    s = _mm_add_ps(s, _mm_movehl_ps(s, s))
    s = _mm_add_ss(s, _mm_movehdup_ps(s))
    return _mm_cvtss_f32(s)


@cython_benchmark(syntax="cy_simd", args=(1000000,))
def layer_norm(int n):
    """f32 layer norm with AVX2+FMA — 4-accumulator rsum, register hsum."""
    cdef int group_size = 1024  # larger groups to exercise SIMD
    cdef int num_groups = n // group_size
    cdef float eps = 1e-5

    cdef float *vals = <float *>malloc(group_size * sizeof(float))
    cdef float *out = <float *>malloc(group_size * sizeof(float))
    if not vals or not out:
        raise MemoryError()

    cdef int g, i, offset
    cdef int end32 = (group_size // 32) * 32
    cdef int end8 = (group_size // 8) * 8
    cdef double total = 0.0
    cdef float mean, var_sum, inv_std
    cdef __m256 vs0, vs1, vs2, vs3, vmean, vd, vinv

    for g in range(num_groups):
        offset = g * group_size
        for i in range(group_size):
            vals[i] = (((offset + i) * 17 + 5) % 1000) / 10.0

        # Mean: 4-accumulator
        vs0 = _mm256_setzero_ps()
        vs1 = _mm256_setzero_ps()
        vs2 = _mm256_setzero_ps()
        vs3 = _mm256_setzero_ps()
        for i in range(0, end32, 32):
            vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&vals[i]))
            vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(&vals[i + 8]))
            vs2 = _mm256_add_ps(vs2, _mm256_loadu_ps(&vals[i + 16]))
            vs3 = _mm256_add_ps(vs3, _mm256_loadu_ps(&vals[i + 24]))
        for i in range(end32, end8, 8):
            vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&vals[i]))
        vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
        mean = _hsum_avx(vs0)
        for i in range(end8, group_size):
            mean += vals[i]
        mean /= group_size

        # Variance: FMA
        vmean = _mm256_set1_ps(mean)
        vs0 = _mm256_setzero_ps()
        vs1 = _mm256_setzero_ps()
        vs2 = _mm256_setzero_ps()
        vs3 = _mm256_setzero_ps()
        for i in range(0, end32, 32):
            vd = _mm256_sub_ps(_mm256_loadu_ps(&vals[i]), vmean)
            vs0 = _mm256_fmadd_ps(vd, vd, vs0)
            vd = _mm256_sub_ps(_mm256_loadu_ps(&vals[i + 8]), vmean)
            vs1 = _mm256_fmadd_ps(vd, vd, vs1)
            vd = _mm256_sub_ps(_mm256_loadu_ps(&vals[i + 16]), vmean)
            vs2 = _mm256_fmadd_ps(vd, vd, vs2)
            vd = _mm256_sub_ps(_mm256_loadu_ps(&vals[i + 24]), vmean)
            vs3 = _mm256_fmadd_ps(vd, vd, vs3)
        for i in range(end32, end8, 8):
            vd = _mm256_sub_ps(_mm256_loadu_ps(&vals[i]), vmean)
            vs0 = _mm256_fmadd_ps(vd, vd, vs0)
        vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
        var_sum = _hsum_avx(vs0)
        for i in range(end8, group_size):
            var_sum += (vals[i] - mean) * (vals[i] - mean)

        # Normalize
        inv_std = 1.0 / sqrtf(var_sum / group_size + eps)
        vinv = _mm256_set1_ps(inv_std)
        for i in range(0, end8, 8):
            vd = _mm256_sub_ps(_mm256_loadu_ps(&vals[i]), vmean)
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(vd, vinv))
        for i in range(end8, group_size):
            out[i] = (vals[i] - mean) * inv_std

        # Accumulate output
        vs0 = _mm256_setzero_ps()
        for i in range(0, end8, 8):
            vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&out[i]))
        total += <double>_hsum_avx(vs0)
        for i in range(end8, group_size):
            total += out[i]

    free(vals)
    free(out)
    return total
