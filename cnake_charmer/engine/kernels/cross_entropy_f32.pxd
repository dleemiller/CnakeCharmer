# Cross entropy f32 kernel declarations.

cdef double cross_entropy_f32(const float *logits, int n, int target) noexcept nogil
cdef double cross_entropy_f32_avx(const float *logits, int n, int target) noexcept nogil
