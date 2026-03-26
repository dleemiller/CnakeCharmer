# Attention scores f32 kernel declarations.
# scores[i][j] = dot(Q[i], K[j]) / sqrt(d_model)

cdef void attention_scores_f32(const float *Q, const float *K, float *scores,
                               int seq_len, int d_model) noexcept nogil
cdef void attention_scores_f32_avx(const float *Q, const float *K, float *scores,
                                   int seq_len, int d_model) noexcept nogil
