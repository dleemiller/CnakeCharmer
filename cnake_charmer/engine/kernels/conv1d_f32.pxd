# Conv1D f32 kernel declarations.
# output[i] = sum(input[i+k] * kernel[k]) for k in [0, kernel_size).

cdef void conv1d_f32(const float *inp, const float *kernel,
                     float *out, int n, int kernel_size) noexcept nogil
cdef void conv1d_f32_avx(const float *inp, const float *kernel,
                         float *out, int n, int kernel_size) noexcept nogil
