# Depthwise conv 1D f32 kernel declarations.

cdef void depthwise_conv_f32(const float *inp, const float *kernel, float *out,
                             int channels, int spatial, int kernel_size) noexcept nogil
cdef void depthwise_conv_f32_avx(const float *inp, const float *kernel, float *out,
                                  int channels, int spatial, int kernel_size) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
