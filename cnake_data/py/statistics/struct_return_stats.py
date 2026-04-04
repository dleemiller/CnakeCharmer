"""Descriptive statistics with struct return pattern.

Computes mean, variance, and skewness of a hash-derived
array, returning all three in a struct (dict in Python).

Keywords: statistics, descriptive, struct return, skewness, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def struct_return_stats(n: int) -> float:
    """Compute descriptive stats, return weighted sum.

    Args:
        n: Number of elements.

    Returns:
        mean + variance + abs(skewness).
    """
    mask = 0xFFFFFFFF

    # Pass 1: compute mean
    s = 0.0
    for i in range(n):
        h = ((i * 2654435761) & mask) ^ ((i * 2246822519) & mask)
        val = (h & 0xFFFF) / 65535.0
        s += val
    mean = s / n

    # Pass 2: variance and skewness
    var_sum = 0.0
    skew_sum = 0.0
    for i in range(n):
        h = ((i * 2654435761) & mask) ^ ((i * 2246822519) & mask)
        val = (h & 0xFFFF) / 65535.0
        diff = val - mean
        var_sum += diff * diff
        skew_sum += diff * diff * diff

    variance = var_sum / n
    if variance > 0:
        std = variance**0.5
        skewness = (skew_sum / n) / (std * std * std)
    else:
        skewness = 0.0

    return mean + variance + abs(skewness)
