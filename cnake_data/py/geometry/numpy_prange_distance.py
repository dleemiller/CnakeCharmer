"""Pairwise Euclidean distance sum for 3D points using NumPy.

Computes the sum of all pairwise distances between n points.

Keywords: geometry, distance, pairwise, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def numpy_prange_distance(n: int) -> float:
    """Sum all pairwise Euclidean distances for n 3D points.

    Args:
        n: Number of 3D points.

    Returns:
        Total sum of all pairwise distances.
    """
    rng = np.random.RandomState(42)
    points = rng.standard_normal((n, 3))
    total = 0.0
    for i in range(n):
        diff = points[i + 1 :] - points[i]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        total += float(np.sum(dists))
    return total
