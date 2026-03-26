# Instance normalization f32 kernel declarations.

cdef void instance_norm_f32(const float *inp, float *out, int channels, int spatial, float eps) noexcept nogil
cdef void instance_norm_f32_avx(const float *inp, float *out, int channels, int spatial, float eps) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
