# Sigmoid f32 kernel declarations.
# sigmoid(x) = 1 / (1 + exp(-x))

cdef void sigmoid_f32(const float *inp, float *out, int n) noexcept nogil
cdef void sigmoid_f32_avx(const float *inp, float *out, int n) noexcept nogil
