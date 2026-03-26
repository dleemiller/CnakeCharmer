# SiLU f32 kernel declarations.

cdef void silu_f32(const float *inp, float *out, int n) noexcept nogil
cdef void silu_f32_avx(const float *inp, float *out, int n) noexcept nogil
cdef double reduce_sum_f32(const float *data, int n) noexcept nogil
cdef double reduce_sum_f32_avx(const float *data, int n) noexcept nogil
