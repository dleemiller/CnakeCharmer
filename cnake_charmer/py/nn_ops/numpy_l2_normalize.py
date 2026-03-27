"""L2-normalize rows of a matrix using NumPy.

Normalizes each row to unit length using vectorized NumPy operations.

Keywords: nn_ops, L2, normalize, numpy, benchmark
"""

import numpy as np

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def numpy_l2_normalize(n: int) -> float:
    """L2-normalize rows of an n x 256 matrix and return sum.

    Args:
        n: Number of rows.

    Returns:
        Sum of all elements in the normalized matrix.
    """
    rng = np.random.RandomState(42)
    mat = rng.standard_normal((n, 256))
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    result = mat / norms
    return float(np.sum(result))
