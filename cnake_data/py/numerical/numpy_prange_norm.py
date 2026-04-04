"""Row-wise L2 norm of a 2D NumPy array.

Uses np.linalg.norm with axis=1 for vectorized computation.

Keywords: numerical, norm, L2, rows, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def numpy_prange_norm(n: int) -> float:
    """Compute row-wise L2 norms of an n x 1000 matrix.

    Args:
        n: Number of rows.

    Returns:
        Sum of all row norms.
    """
    rng = np.random.RandomState(42)
    mat = rng.standard_normal((n, 1000))
    norms = np.linalg.norm(mat, axis=1)
    return float(np.sum(norms))
