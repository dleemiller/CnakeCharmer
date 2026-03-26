# Embedding lookup f32 kernel declarations.

cdef double embedding_lookup_f32(const float *table, int vocab_size, int dim, int n) noexcept nogil
cdef double embedding_lookup_f32_avx(const float *table, int vocab_size, int dim, int n) noexcept nogil
