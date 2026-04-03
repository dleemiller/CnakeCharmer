"""Gaussian process squared exponential kernel matrix.

Keywords: gaussian process, kernel, covariance, rbf, squared exponential
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def gp_kernel(n):
    """Compute GP covariance matrix for n data points in 5 dimensions.

    Uses isotropic squared exponential kernel:
    K[i,j] = sigma * exp(-0.5 * ||x_i - x_j||^2 / tau^2) + eps * delta(i,j)

    Args:
        n: Number of data points.

    Returns:
        Tuple of (trace, off_diag_sum, max_off_diag).
    """
    d = 5
    sigma = 1.0
    tau = 0.5
    epsilon = 1e-6
    inv_2tau2 = 1.0 / (2.0 * tau * tau)

    # Generate deterministic input features
    X = []
    for i in range(n):
        row = []
        for k in range(d):
            row.append(((i * 7 + k * 13 + 3) % 97) / 97.0)
        X.append(row)

    # Compute kernel matrix (upper triangle + diagonal)
    trace = 0.0
    off_diag_sum = 0.0
    max_off_diag = 0.0

    for i in range(n):
        trace += sigma + epsilon

        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(d):
                diff = X[i][k] - X[j][k]
                dist_sq += diff * diff
            val = sigma * math.exp(-dist_sq * inv_2tau2)
            off_diag_sum += val
            if val > max_off_diag:
                max_off_diag = val

    return (trace, off_diag_sum, max_off_diag)
