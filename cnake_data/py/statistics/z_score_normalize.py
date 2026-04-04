"""Z-score normalize a deterministic dataset.

Keywords: statistics, z-score, normalization, standardize, mean, stddev, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def z_score_normalize(n: int) -> tuple:
    """Z-score normalize a dataset and return summary statistics.

    Data: v[i] = cos(i * 0.007) * 30.0 + (i * 23 + 3) % 151

    Two-pass: compute mean and stddev, then normalize and summarize.

    Args:
        n: Number of data points.

    Returns:
        Tuple of (sum_of_normalized, max_normalized, min_normalized).
    """
    # Pass 1: compute mean
    total = 0.0
    for i in range(n):
        total += math.cos(i * 0.007) * 30.0 + (i * 23 + 3) % 151
    mean = total / n

    # Pass 2: compute variance
    var_sum = 0.0
    for i in range(n):
        val = math.cos(i * 0.007) * 30.0 + (i * 23 + 3) % 151
        d = val - mean
        var_sum += d * d
    stddev = math.sqrt(var_sum / n)

    if stddev == 0.0:
        return (0.0, 0.0, 0.0)

    # Pass 3: normalize and compute summary
    norm_sum = 0.0
    norm_max = -1e308
    norm_min = 1e308
    for i in range(n):
        val = math.cos(i * 0.007) * 30.0 + (i * 23 + 3) % 151
        z = (val - mean) / stddev
        norm_sum += z
        if z > norm_max:
            norm_max = z
        if z < norm_min:
            norm_min = z

    return (norm_sum, norm_max, norm_min)
