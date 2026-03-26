# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sigmoid f32 kernels — scalar and XNNPACK-style polynomial AVX2.

scalar: 1/(1+exp(-x)) using libc expf
AVX2: degree-5 polynomial exp approximation (no expf calls),
      following XNNPACK f32-vsigmoid-avx-rr2-p5 pattern.
"""

from libc.math cimport expf

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
    __m256 _mm256_insertf128_ps(__m256 a, __m128 b, int imm8) noexcept
    __m128 _mm256_extractf128_ps(__m256 a, int imm8) noexcept

    # SSE2
    ctypedef int __m128i
    __m128i _mm_castps_si128(__m128 a) noexcept
    __m128 _mm_castsi128_ps(__m128i a) noexcept
    __m128i _mm_slli_epi32(__m128i a, int imm8) noexcept
    __m256 _mm256_castps128_ps256(__m128 a) noexcept


cdef void sigmoid_f32(const float *inp, float *out, int n) noexcept nogil:
    """Scalar sigmoid: out[i] = 1 / (1 + exp(-inp[i]))."""
    cdef int i
    for i in range(n):
        out[i] = 1.0 / (1.0 + expf(-inp[i]))


cdef void sigmoid_f32_avx(const float *inp, float *out, int n) noexcept nogil:
    """XNNPACK-style AVX sigmoid using degree-5 polynomial exp approximation.

    No expf() calls — entire computation in SIMD registers.
    Follows f32-vsigmoid-avx-rr2-p5-div pattern.
    """
    # Constants from XNNPACK
    cdef __m256 vsign_mask = _mm256_set1_ps(-0.0)
    cdef __m256 vlog2e = _mm256_set1_ps(1.4426950216293335)      # 0x1.715476p0
    cdef __m256 vmagic_bias = _mm256_set1_ps(12582912.5)         # 0x1.8000FEp23
    cdef __m256 vminus_ln2_hi = _mm256_set1_ps(-0.6931457519531250) # -0x1.62E400p-1
    cdef __m256 vminus_ln2_lo = _mm256_set1_ps(-1.4286067653e-6)    # -0x1.7F7D1Cp-20
    cdef __m256 vc5 = _mm256_set1_ps(0.008513249971270561)       # 0x1.0F9F9Cp-7
    cdef __m256 vc4 = _mm256_set1_ps(0.04166689515113831)        # 0x1.573A1Ap-5
    cdef __m256 vc3 = _mm256_set1_ps(0.16666664183139801)        # 0x1.555A80p-3
    cdef __m256 vc2 = _mm256_set1_ps(0.49999985098838806)        # 0x1.FFFDC6p-2
    cdef __m256 vc1 = _mm256_set1_ps(0.9999998807907104)         # 0x1.FFFFF6p-1
    cdef __m256 vone = _mm256_set1_ps(1.0)
    cdef __m256 vdenorm_cutoff = _mm256_set1_ps(-87.33654022216797) # -0x1.5D589Ep+6

    cdef int i
    cdef int end8 = (n // 8) * 8
    cdef __m256 vx, vz, vn, vs, vt, vp, ve, vd, vf
    cdef __m128 vs_lo, vs_hi

    for i in range(0, end8, 8):
        vx = _mm256_loadu_ps(&inp[i])

        # z = |x| (negate for exp(-|x|))
        vz = _mm256_or_ps(vx, vsign_mask)

        # n = round(z * log2e)  (with magic bias for rounding)
        vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias)

        # s = 2^n via bit manipulation (extract integer part, shift to exponent)
        vs_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23))
        vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 23))
        vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1)

        # Remove magic bias to get n as float
        vn = _mm256_sub_ps(vn, vmagic_bias)

        # t = z - n*ln2  (range reduction, 2 steps for precision)
        vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz)
        vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt)

        # p(t) = c5*t + c4, then Horner's method
        vp = _mm256_add_ps(_mm256_mul_ps(vc5, vt), vc4)
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3)
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2)
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc1)

        # e = s * (t*p + 1) = exp(z) approximation
        vt = _mm256_mul_ps(vt, vs)
        ve = _mm256_add_ps(_mm256_mul_ps(vt, vp), vs)

        # sigmoid = e / (e + 1)
        vd = _mm256_add_ps(ve, vone)
        vf = _mm256_div_ps(ve, vd)

        # Flush denorms to zero for very negative inputs
        vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, 1), vf)  # _CMP_LT_OS = 1

        # For positive x: sigmoid(x) = 1 - sigmoid(-x)
        vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx)

        _mm256_storeu_ps(&out[i], vf)

    # Scalar remainder
    for i in range(end8, n):
        out[i] = 1.0 / (1.0 + expf(-inp[i]))
