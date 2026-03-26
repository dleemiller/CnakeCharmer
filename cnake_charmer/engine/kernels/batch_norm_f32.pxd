# Batch norm f32 kernel declarations (inference mode).
# output[i] = gamma * (input[i] - mean) * inv_std + beta

cdef void batch_norm_f32(const float *inp, float *out, int n,
                         float mean, float inv_std,
                         float gamma, float beta) noexcept nogil
cdef void batch_norm_f32_avx(const float *inp, float *out, int n,
                             float mean, float inv_std,
                             float gamma, float beta) noexcept nogil
