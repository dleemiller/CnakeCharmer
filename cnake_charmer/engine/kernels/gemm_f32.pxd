# GEMM f32 kernel declarations.
# C = A * B where A is MxK, B is KxN, C is MxN (all row-major flat arrays).

cdef void gemm_f32(const float *A, const float *B, float *C,
                   int M, int N, int K) noexcept nogil

cdef void gemm_f32_avx(const float *A, const float *B, float *C,
                       int M, int N, int K) noexcept nogil
