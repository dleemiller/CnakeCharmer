# Layer norm f32 kernel declarations.
# Two-pass normalization: mean, variance, then normalize.

cdef void layer_norm_f32(const float *inp, float *out,
                         int n, float epsilon) noexcept nogil
cdef void layer_norm_f32_avx(const float *inp, float *out,
                             int n, float epsilon) noexcept nogil
