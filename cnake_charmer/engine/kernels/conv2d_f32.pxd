# Conv2D f32 kernel declarations.
# 2D convolution on flat row-major arrays.

cdef void conv2d_f32(const float *inp, const float *kernel, float *out,
                     int in_h, int in_w, int kh, int kw) noexcept nogil
cdef void conv2d_f32_avx(const float *inp, const float *kernel, float *out,
                         int in_h, int in_w, int kh, int kw) noexcept nogil
