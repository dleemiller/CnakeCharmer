"""Welford's online algorithm for population variance.

Keywords: statistics, variance, Welford, online algorithm, mean, LCG
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def welford_variance(n: int) -> tuple[float, float]:
    """Compute mean and population variance using Welford's online algorithm.

    Generates n values via a linear congruential generator:
      x[i] = ((i * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296.0

    Uses Welford's update:
      delta = x - mean
      mean += delta / (i + 1)
      M2 += delta * (x - mean)
    Variance = M2 / n

    Args:
        n: Number of values.

    Returns:
        (mean, variance) as floats.
    """
    mean = 0.0
    M2 = 0.0

    for i in range(n):
        x = ((i * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296.0
        delta = x - mean
        mean += delta / (i + 1)
        M2 += delta * (x - mean)

    variance = M2 / n
    return (mean, variance)
