"""Weighted Euclidean distance using NumPy.

Keywords: statistics, weighted distance, euclidean, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def pythran_weighted_dist(n: int) -> float:
    """Compute weighted Euclidean distances between row pairs and return sum.

    Reshapes n elements into rows of 500 columns. For consecutive row pairs,
    compute sqrt(sum(w * (a - b)**2)). NumPy creates multiple temporaries.

    Args:
        n: Total element count (reshaped as n//500 rows of 500 cols).

    Returns:
        Sum of all weighted distances.
    """
    rng = np.random.RandomState(42)
    cols = 500
    rows = n // cols
    if rows < 2:
        return 0.0
    mat = rng.standard_normal((rows, cols))
    w = np.abs(rng.standard_normal(cols))
    w = w / np.sum(w)  # normalize weights

    total = 0.0
    for i in range(rows - 1):
        diff = mat[i] - mat[i + 1]
        total += float(np.sqrt(np.sum(w * diff * diff)))
    return total
