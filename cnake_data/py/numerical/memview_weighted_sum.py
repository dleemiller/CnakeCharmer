"""Weighted prefix sum with exponential decay.

Keywords: weighted sum, prefix sum, exponential decay, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def memview_weighted_sum(n: int) -> float:
    """Compute a weighted prefix sum where weights decay exponentially.

    For each position i, computes sum of values[j] * decay^(i-j) for j <= i.
    Returns the sum of all prefix sums.

    Args:
        n: Array length.

    Returns:
        Sum of all weighted prefix sums.
    """
    decay = 0.999
    values = [0.0] * n
    result = [0.0] * n

    for i in range(n):
        values[i] = ((i * 2654435761) % 1000) / 100.0

    # Compute using recurrence: result[i] = values[i] + decay * result[i-1]
    result[0] = values[0]
    for i in range(1, n):
        result[i] = values[i] + decay * result[i - 1]

    total = 0.0
    for i in range(n):
        total += result[i]

    return total
