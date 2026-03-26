# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sigmoid activation — XNNPACK AVX2 rr1-p5-div polynomial.

Matches xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div from XNNPACK.
Uses FMA throughout + 256-bit integer shift (AVX2 required).

Keywords: sigmoid, activation, neural network, elementwise, f32, simd, avx, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport expf
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    ctypedef int __m256i
    ctypedef float __m128

    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_set1_ps(float a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_sub_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_mul_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    __m256 _mm256_div_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_or_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_andnot_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 mask) noexcept
    __m256 _mm256_cmp_ps(__m256 a, __m256 b, int imm8) noexcept
    __m256 _mm256_setzero_ps() noexcept
    __m256 _mm256_castsi256_ps(__m256i a) noexcept
    __m256i _mm256_castps_si256(__m256 a) noexcept
    __m256i _mm256_slli_epi32(__m256i a, int imm8) noexcept
    __m128 _mm256_castps256_ps128(__m256 a) noexcept
    __m128 _mm256_extractf128_ps(__m256 a, int imm8) noexcept
    __m128 _mm_add_ps(__m128 a, __m128 b) noexcept
    __m128 _mm_add_ss(__m128 a, __m128 b) noexcept
    __m128 _mm_movehl_ps(__m128 a, __m128 b) noexcept
    __m128 _mm_movehdup_ps(__m128 a) noexcept
    float _mm_cvtss_f32(__m128 a) noexcept


cdef inline float _hsum_avx(__m256 v) noexcept nogil:
    cdef __m128 lo = _mm256_castps256_ps128(v)
    cdef __m128 hi = _mm256_extractf128_ps(v, 1)
    cdef __m128 s = _mm_add_ps(lo, hi)
    s = _mm_add_ps(s, _mm_movehl_ps(s, s))
    s = _mm_add_ss(s, _mm_movehdup_ps(s))
    return _mm_cvtss_f32(s)


@cython_benchmark(syntax="cy_simd", args=(1000000,))
def sigmoid(int n):
    """f32 sigmoid — XNNPACK AVX2 rr1-p5-div polynomial."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not data or not out:
        raise MemoryError()

    # Cheap input gen (same pattern as cy version)
    cdef int i
    for i in range(n):
        data[i] = <float>(((i * 17 + 5) % 1000) / 100.0 - 5.0)

    # XNNPACK AVX2 rr1-p5-div constants (exact hex float values)
    cdef __m256 vsign_mask = _mm256_set1_ps(-0.0)
    cdef __m256 vmagic_bias = _mm256_set1_ps(12583039.0)                  # 0x1.8000FEp23
    cdef __m256 vlog2e = _mm256_set1_ps(1.44269502162933349609)           # 0x1.715476p0
    cdef __m256 vminus_ln2 = _mm256_set1_ps(-0.693147182464599609375)     # -0x1.62E430p-1
    cdef __m256 vc5 = _mm256_set1_ps(0.00828929059207439422607)           # 0x1.0F9F9Cp-7
    cdef __m256 vc4 = _mm256_set1_ps(0.0418978221714496612549)            # 0x1.573A1Ap-5
    cdef __m256 vc3 = _mm256_set1_ps(0.166676521301269531250)             # 0x1.555A80p-3
    cdef __m256 vc2 = _mm256_set1_ps(0.499991506338119506836)             # 0x1.FFFDC6p-2
    cdef __m256 vc1 = _mm256_set1_ps(0.999999701976776123047)             # 0x1.FFFFF6p-1
    cdef __m256 vone = _mm256_set1_ps(1.0)
    cdef __m256 vdenorm_cutoff = _mm256_set1_ps(-87.3365402221679687500)  # -0x1.5D589Ep+6

    cdef int end8 = (n // 8) * 8
    cdef __m256 vx, vz, vn, vs, vt, vp, ve, vd, vf

    for i in range(0, end8, 8):
        vx = _mm256_loadu_ps(&data[i])
        vz = _mm256_or_ps(vx, vsign_mask)
        vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias)
        vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23))
        vn = _mm256_sub_ps(vn, vmagic_bias)
        vt = _mm256_fmadd_ps(vn, vminus_ln2, vz)
        vp = _mm256_fmadd_ps(vc5, vt, vc4)
        vp = _mm256_fmadd_ps(vp, vt, vc3)
        vp = _mm256_fmadd_ps(vp, vt, vc2)
        vp = _mm256_fmadd_ps(vp, vt, vc1)
        vt = _mm256_mul_ps(vt, vs)
        ve = _mm256_fmadd_ps(vt, vp, vs)
        vd = _mm256_add_ps(ve, vone)
        vf = _mm256_div_ps(ve, vd)
        vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, 1), vf)
        vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx)
        _mm256_storeu_ps(&out[i], vf)

    for i in range(end8, n):
        out[i] = 1.0 / (1.0 + expf(-data[i]))

    # Reduce with AVX
    cdef int rend16 = (n // 16) * 16
    cdef __m256 acc0 = _mm256_setzero_ps()
    cdef __m256 acc1 = _mm256_setzero_ps()
    for i in range(0, rend16, 16):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&out[i + 8]))
    for i in range(rend16, end8, 8):
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&out[i]))
    acc0 = _mm256_add_ps(acc0, acc1)
    cdef double total = <double>_hsum_avx(acc0)
    for i in range(end8, n):
        total += out[i]

    free(data)
    free(out)
    return total
