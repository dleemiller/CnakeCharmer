# ReLU f32 kernel declarations.
# These can be cimported by benchmark wrappers or a future engine.

cdef void relu_f32(const float *inp, float *out, int n) noexcept nogil
cdef void relu_f32_avx(const float *inp, float *out, int n) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
cdef double reduce_sum_f32_avx(const float *data, int n) noexcept nogil
