# Max pool 1D f32 kernel declarations.
# Max pooling with kernel_size and stride.

cdef void max_pool_1d_f32(const float *inp, float *out,
                          int n, int kernel_size, int stride) noexcept nogil
cdef void max_pool_1d_f32_avx(const float *inp, float *out,
                              int n, int kernel_size, int stride) noexcept nogil
