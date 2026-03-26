# Elementwise add f32 kernel declarations.
# output[i] = a[i] + b[i]

cdef void elementwise_add_f32(const float *a, const float *b,
                              float *out, int n) noexcept nogil
cdef void elementwise_add_f32_avx(const float *a, const float *b,
                                  float *out, int n) noexcept nogil
