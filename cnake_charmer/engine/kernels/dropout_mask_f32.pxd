# Dropout mask f32 kernel declarations.

cdef void dropout_mask_f32(const float *inp, float *out, int n, float p) noexcept nogil
cdef void dropout_mask_f32_avx(const float *inp, float *out, int n, float p) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
