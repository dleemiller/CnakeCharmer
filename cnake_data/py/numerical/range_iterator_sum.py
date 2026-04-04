"""Sum values from a custom range-like iterator with stride and filter.

Keywords: iterator, range, sum, generator, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def range_iterator_sum(n: int) -> float:
    """Create a stepped range [0, n) with step 3, filter to only values where hash < 128.

    Iterates through all matching values and computes their weighted sum.

    Args:
        n: Upper bound of range.

    Returns:
        Weighted sum of filtered values.
    """
    total = 0.0
    count = 0
    val = 0
    while val < n:
        h = ((val * 2654435761 + 7) >> 8) & 0xFF
        if h < 128:
            total += val * (1.0 + (count % 10) * 0.1)
            count += 1
        val += 3

    return total
