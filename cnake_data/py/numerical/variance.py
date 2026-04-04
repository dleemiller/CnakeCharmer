"""
Compute population variance of a deterministic sequence using two-pass algorithm.

Keywords: numerical, variance, statistics, two-pass, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def variance(n: int) -> float:
    """Compute population variance of sequence value[i] = (i*17+5) % 1000 / 10.0.

    Uses a two-pass algorithm: first compute the mean, then sum the squared
    deviations from the mean and divide by n.

    Args:
        n: Length of the sequence.

    Returns:
        Population variance as a float.
    """
    # First pass: compute mean
    total = 0.0
    for i in range(n):
        total += (i * 17 + 5) % 1000 / 10.0
    mean = total / n

    # Second pass: sum of squared deviations
    var_sum = 0.0
    for i in range(n):
        diff = (i * 17 + 5) % 1000 / 10.0 - mean
        var_sum += diff * diff

    return var_sum / n
