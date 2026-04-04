"""Compute excess kurtosis and skewness of a deterministic dataset.

Keywords: statistics, kurtosis, skewness, moments, distribution, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def kurtosis(n: int) -> tuple:
    """Compute excess kurtosis, skewness, and mean of a deterministic dataset.

    Data: v[i] = sin(i * 0.01) * 50.0 + (i * 13 + 7) % 97

    Two-pass algorithm: first compute mean, then central moments.

    Args:
        n: Number of data points.

    Returns:
        Tuple of (excess_kurtosis, skewness, mean).
    """
    # Pass 1: compute mean
    total = 0.0
    for i in range(n):
        total += math.sin(i * 0.01) * 50.0 + (i * 13 + 7) % 97
    mean = total / n

    # Pass 2: compute central moments
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        val = math.sin(i * 0.01) * 50.0 + (i * 13 + 7) % 97
        d = val - mean
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2

    m2 /= n
    m3 /= n
    m4 /= n

    if m2 == 0.0:
        return (0.0, 0.0, mean)

    # Excess kurtosis = m4/m2^2 - 3
    excess_kurt = m4 / (m2 * m2) - 3.0
    # Skewness = m3 / m2^(3/2)
    skew = m3 / (m2 * math.sqrt(m2))

    return (excess_kurt, skew, mean)
