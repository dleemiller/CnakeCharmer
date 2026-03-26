# Average pooling 1D f32 kernel declarations.

cdef void avg_pool_1d_f32(const float *inp, float *out, int n, int kernel, int stride) noexcept nogil
cdef void avg_pool_1d_f32_avx(const float *inp, float *out, int n, int kernel, int stride) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
cdef double reduce_sum_f32_avx(const float *data, int n) noexcept nogil
