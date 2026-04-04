"""Parallel sum of squares of deterministic values.

Keywords: numerical, sum, squares, reduction, parallel, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def prange_sum_squares(n: int) -> float:
    """Compute sum of squares of n hash-derived values.

    Args:
        n: Number of values.

    Returns:
        Sum of squares as a float.
    """
    total = 0.0
    for i in range(n):
        val = (i * 2654435761 & 0xFFFFFFFF) / 4294967296.0
        total += val * val

    return total
