"""Compute weighted 50th percentile (median) of n generated values.

Keywords: weighted, percentile, median, statistics, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def weighted_percentile(n: int) -> float:
    """Compute the weighted 50th percentile of n values.

    Values: v[i] = (i*7+3) % 1000. Weights: w[i] = (i*13+7) % 10 + 1.
    Uses sorting + cumulative weight scan to find the median.

    Args:
        n: Number of data points.

    Returns:
        Weighted 50th percentile as float.
    """
    # Generate values and weights
    values = [(i * 7 + 3) % 1000 for i in range(n)]
    weights = [(i * 13 + 7) % 10 + 1 for i in range(n)]

    # Sort by value
    indices = list(range(n))
    indices.sort(key=lambda idx: values[idx])

    # Compute total weight
    total_weight = 0
    for i in range(n):
        total_weight += weights[i]

    # Find the 50th percentile: first value where cumulative weight >= 50% of total
    threshold = total_weight * 0.5
    cumulative = 0
    for i in range(n):
        idx = indices[i]
        cumulative += weights[idx]
        if cumulative >= threshold:
            return float(values[idx])

    return float(values[indices[n - 1]])
