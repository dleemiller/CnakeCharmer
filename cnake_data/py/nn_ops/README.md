# nn_ops — Neural Network Operations

Neural-network-oriented problem pairs used for Python -> Cython training.

## Implemented Operations

### Core
- `conv1d`
- `conv2d`
- `gemm`
- `relu`
- `sigmoid`
- `gelu`
- `softmax_stable`
- `batch_norm`
- `cross_entropy`

### Pooling
- `max_pool_1d`
- `avg_pool_1d`
- `global_avg_pool`

### Normalization
- `layer_norm`
- `instance_norm`
- `numpy_l2_normalize`
- `numpy_batch_norm`

### Elementwise / Composition
- `elementwise_add`
- `elementwise_mul`
- `residual_add`
- `dropout_mask`
- `silu`

### Attention / Embedding
- `attention_scores`
- `embedding_lookup`

### Specialized / Interop Variants
- `depthwise_conv`
- `numpy_softmax`
- `numpy_cross_entropy`
- `ufunc_fused_sigmoid`

## Notes

- These files are benchmark-decorated problem definitions in `cnake_data/py/nn_ops`.
- Corresponding Cython implementations live in `cnake_data/cy/nn_ops`.
- Benchmarks are generated via `run_benchmarks.py` / `make benchmark`.
