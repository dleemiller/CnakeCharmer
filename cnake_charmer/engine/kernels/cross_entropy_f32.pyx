# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cross entropy f32 kernels -- scalar and AVX2 implementations.

Cross entropy loss: -logits[target] + log(sum(exp(logits - max_logit))).
Pure compute, no allocation.
"""

from libc.math cimport exp, log

cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    void _mm256_storeu_ps(float *mem, __m256 a) noexcept
    __m256 _mm256_add_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_max_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_setzero_ps() noexcept


cdef double cross_entropy_f32(const float *logits, int n, int target) noexcept nogil:
    """Scalar cross entropy loss."""
    cdef int i
    cdef float max_logit = logits[0]
    cdef double sum_exp = 0.0

    # Find max
    for i in range(1, n):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Sum exp(logits - max)
    for i in range(n):
        sum_exp += exp(<double>(logits[i] - max_logit))

    # Loss = -logits[target] + log(sum_exp) + max_logit
    return -<double>logits[target] + log(sum_exp) + <double>max_logit


cdef double cross_entropy_f32_avx(const float *logits, int n, int target) noexcept nogil:
    """AVX2 cross entropy -- scalar fallback (exp has no AVX intrinsic)."""
    return cross_entropy_f32(logits, n, target)
