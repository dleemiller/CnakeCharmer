# Residual add + ReLU f32 kernel declarations.

cdef void residual_add_f32(const float *inp, const float *residual, float *out, int n) noexcept nogil
cdef void residual_add_f32_avx(const float *inp, const float *residual, float *out, int n) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
cdef double reduce_sum_f32_avx(const float *data, int n) noexcept nogil
