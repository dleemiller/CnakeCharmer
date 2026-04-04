"""Find argmax of each row in a 2D NumPy array.

Uses np.argmax with axis=1 for vectorized row-wise argmax.

Keywords: numerical, argmax, rows, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def numpy_argmax_rows(n: int) -> int:
    """Find argmax of each row in an n x 512 matrix.

    Args:
        n: Number of rows.

    Returns:
        Sum of all argmax indices.
    """
    rng = np.random.RandomState(42)
    mat = rng.standard_normal((n, 512))
    indices = np.argmax(mat, axis=1)
    return int(np.sum(indices))
