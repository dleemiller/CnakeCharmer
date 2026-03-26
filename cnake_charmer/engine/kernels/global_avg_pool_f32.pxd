# Global average pool f32 kernel declarations.

cdef void global_avg_pool_f32(const float *inp, float *out, int channels, int spatial) noexcept nogil
cdef void global_avg_pool_f32_avx(const float *inp, float *out, int channels, int spatial) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
