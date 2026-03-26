# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Instance normalization — AVX2+FMA with 4-accumulator rsum pattern.

Keywords: instance_norm, normalization, neural network, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
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
def instance_norm(int n):
    """f32 instance norm with AVX2+FMA — 4-accumulator rsum per channel."""
    cdef int channels = 16
    cdef int spatial = n // channels
    cdef float eps = 1e-5

    cdef float *data = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not data or not out:
        raise MemoryError()

    cdef int c, s, offset
    cdef float mean, var_sum, inv_std
    cdef double total = 0.0
    cdef int end32 = (spatial // 32) * 32
    cdef int end8 = (spatial // 8) * 8
    cdef __m256 vs0, vs1, vs2, vs3, vmean, vd, vinv

    # Generate input
    for c in range(n):
        data[c] = sin(c * 0.01) * 10.0

    for c in range(channels):
        offset = c * spatial

        # Mean: 4-accumulator
        vs0 = _mm256_setzero_ps()
        vs1 = _mm256_setzero_ps()
        vs2 = _mm256_setzero_ps()
        vs3 = _mm256_setzero_ps()
        for s in range(0, end32, 32):
            vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&data[offset + s]))
            vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(&data[offset + s + 8]))
            vs2 = _mm256_add_ps(vs2, _mm256_loadu_ps(&data[offset + s + 16]))
            vs3 = _mm256_add_ps(vs3, _mm256_loadu_ps(&data[offset + s + 24]))
        for s in range(end32, end8, 8):
            vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&data[offset + s]))
        vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
        mean = _hsum_avx(vs0)
        for s in range(end8, spatial):
            mean += data[offset + s]
        mean /= spatial

        # Variance: FMA
        vmean = _mm256_set1_ps(mean)
        vs0 = _mm256_setzero_ps()
        vs1 = _mm256_setzero_ps()
        vs2 = _mm256_setzero_ps()
        vs3 = _mm256_setzero_ps()
        for s in range(0, end32, 32):
            vd = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s]), vmean)
            vs0 = _mm256_fmadd_ps(vd, vd, vs0)
            vd = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s + 8]), vmean)
            vs1 = _mm256_fmadd_ps(vd, vd, vs1)
            vd = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s + 16]), vmean)
            vs2 = _mm256_fmadd_ps(vd, vd, vs2)
            vd = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s + 24]), vmean)
            vs3 = _mm256_fmadd_ps(vd, vd, vs3)
        for s in range(end32, end8, 8):
            vd = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s]), vmean)
            vs0 = _mm256_fmadd_ps(vd, vd, vs0)
        vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3))
        var_sum = _hsum_avx(vs0)
        for s in range(end8, spatial):
            var_sum += (data[offset + s] - mean) * (data[offset + s] - mean)

        # Normalize
        inv_std = 1.0 / sqrtf(var_sum / spatial + eps)
        vinv = _mm256_set1_ps(inv_std)
        for s in range(0, end8, 8):
            vd = _mm256_sub_ps(_mm256_loadu_ps(&data[offset + s]), vmean)
            _mm256_storeu_ps(&out[offset + s], _mm256_mul_ps(vd, vinv))
        for s in range(end8, spatial):
            out[offset + s] = (data[offset + s] - mean) * inv_std

    # Reduce output
    end8 = (n // 8) * 8
    vs0 = _mm256_setzero_ps()
    vs1 = _mm256_setzero_ps()
    cdef int end16 = (n // 16) * 16
    for s in range(0, end16, 16):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&out[s]))
        vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(&out[s + 8]))
    for s in range(end16, end8, 8):
        vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(&out[s]))
    vs0 = _mm256_add_ps(vs0, vs1)
    total = <double>_hsum_avx(vs0)
    for s in range(end8, n):
        total += out[s]

    free(data)
    free(out)
    return total
