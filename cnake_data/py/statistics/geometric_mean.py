"""Geometric mean via log-sum-exp trick to avoid overflow.

Keywords: statistics, geometric mean, log, overflow, numerical, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def geometric_mean(n: int) -> tuple:
    """Compute geometric mean of positive deterministic values using log-sum.

    Data: v[i] = 1.0 + (i * 31 + 11) % 200 / 10.0 + 0.5 * abs(sin(i * 0.03))

    Uses log-sum-exp: geomean = exp(mean(log(v)))

    Args:
        n: Number of data points.

    Returns:
        Tuple of (geometric_mean, log_sum, min_value).
    """
    log_sum = 0.0
    min_val = 1e308
    for i in range(n):
        val = 1.0 + ((i * 31 + 11) % 200) / 10.0 + 0.5 * abs(math.sin(i * 0.03))
        log_sum += math.log(val)
        if val < min_val:
            min_val = val

    log_mean = log_sum / n
    gmean = math.exp(log_mean)

    return (gmean, log_sum, min_val)
