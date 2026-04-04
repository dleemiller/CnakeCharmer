"""Cross-entropy loss between predictions and labels using NumPy.

Computes categorical cross-entropy using vectorized operations.

Keywords: nn_ops, cross entropy, loss, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def numpy_cross_entropy(n: int) -> float:
    """Compute cross-entropy loss and return the scalar value.

    Args:
        n: Number of samples.

    Returns:
        Cross-entropy loss value.
    """
    rng = np.random.RandomState(42)
    raw = rng.standard_normal(n)
    preds = 1.0 / (1.0 + np.exp(-raw))
    preds = np.clip(preds, 1e-12, 1.0 - 1e-12)
    labels = (rng.random(n) > 0.5).astype(np.float64)
    loss = -np.sum(labels * np.log(preds) + (1.0 - labels) * np.log(1.0 - preds))
    return float(loss)
