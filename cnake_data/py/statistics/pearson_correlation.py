"""
Pearson correlation coefficient between two deterministic sequences.

Keywords: statistics, pearson, correlation, numerical, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def pearson_correlation(n: int) -> float:
    """Compute Pearson correlation between two deterministic sequences of length n.

    x[i] = sin(i * 0.1) * 100
    y[i] = sin(i * 0.1 + 0.5) * 100 + (i % 7)

    Two-pass algorithm: compute means, then covariance and standard deviations.

    Args:
        n: Length of the sequences.

    Returns:
        Pearson correlation coefficient.
    """
    # Pass 1: compute means
    sum_x = 0.0
    sum_y = 0.0
    for i in range(n):
        sum_x += math.sin(i * 0.1) * 100.0
        sum_y += math.sin(i * 0.1 + 0.5) * 100.0 + (i % 7)
    mean_x = sum_x / n
    mean_y = sum_y / n

    # Pass 2: compute covariance and standard deviations
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        xi = math.sin(i * 0.1) * 100.0
        yi = math.sin(i * 0.1 + 0.5) * 100.0 + (i % 7)
        dx = xi - mean_x
        dy = yi - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)
