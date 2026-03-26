# nn_ops — Neural Network Operations

Hardware-optimized implementations of common NN primitives.
These go beyond basic `cdef`/`malloc` to use SIMD intrinsics,
OpenMP `prange`, cache blocking, and memory alignment.

## Planned implementations

### Core ops
- [ ] conv1d — 1D convolution with `prange` and optional AVX FMA
- [ ] conv2d — 2D convolution with cache-tiled loops
- [ ] gemm — General matrix multiply with tiling + `prange`
- [ ] relu — Vectorized ReLU using SIMD `_mm256_max_pd`
- [ ] sigmoid — Vectorized sigmoid with fast exp approximation
- [ ] softmax_stable — Numerically stable softmax with `prange`
- [ ] batch_norm — Fused mean/var/normalize in single pass

### Pooling
- [ ] max_pool_1d — Max pooling with stride
- [ ] avg_pool_1d — Average pooling with stride
- [ ] max_pool_2d — 2D max pooling

### Element-wise
- [ ] fused_multiply_add — a*x + b vectorized
- [ ] elementwise_add — Vector addition with SIMD
- [ ] elementwise_exp — Vectorized exp

### Advanced
- [ ] depthwise_conv — Depthwise separable convolution
- [ ] im2col — Image to column transform for GEMM-based conv
- [ ] attention_scores — Scaled dot-product attention Q*K^T/sqrt(d)
- [ ] layer_norm — Layer normalization
- [ ] gelu — GELU activation function

## Cython techniques used

```cython
# OpenMP parallelism
from cython.parallel cimport prange

# AVX intrinsics (x86)
cdef extern from "immintrin.h":
    ctypedef double __m256d
    __m256d _mm256_loadu_pd(double *)
    __m256d _mm256_mul_pd(__m256d, __m256d)
    __m256d _mm256_add_pd(__m256d, __m256d)
    __m256d _mm256_fmadd_pd(__m256d, __m256d, __m256d)
    __m256d _mm256_max_pd(__m256d, __m256d)
    __m256d _mm256_setzero_pd()
    void _mm256_storeu_pd(double *, __m256d)

# Aligned memory allocation
from libc.stdlib cimport aligned_alloc, free
```
