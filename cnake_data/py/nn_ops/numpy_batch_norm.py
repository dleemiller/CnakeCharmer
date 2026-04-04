"""Batch normalization using NumPy vectorized operations.

Applies (x - mean) / sqrt(var + eps) * gamma + beta.

Keywords: nn_ops, batch norm, normalization, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def numpy_batch_norm(n: int) -> float:
    """Apply batch normalization and return sum of result.

    Args:
        n: Number of values.

    Returns:
        Sum of normalized values.
    """
    rng = np.random.RandomState(42)
    data = rng.standard_normal(n)
    gamma = 1.5
    beta = 0.5
    eps = 1e-5
    mean = np.mean(data)
    var = np.var(data)
    result = (data - mean) / np.sqrt(var + eps) * gamma + beta
    return float(np.sum(result))
