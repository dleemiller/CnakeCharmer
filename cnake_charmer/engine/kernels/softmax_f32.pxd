# Softmax f32 kernel declarations.
# Numerically stable softmax: subtract max, exp, normalize.

cdef void softmax_f32(const float *inp, float *out, int n) noexcept nogil
cdef void softmax_f32_avx(const float *inp, float *out, int n) noexcept nogil
