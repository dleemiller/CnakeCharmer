"""
Compute prefix sums of a deterministic array and return the final element.

Keywords: numerical, prefix sum, scan, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def contig_prefix_sum(n: int) -> float:
    """Compute prefix sums and return the last element.

    Input array: data[i] = ((i * 37 + 13) % 1000) / 100.0
    prefix[0] = data[0], prefix[i] = prefix[i-1] + data[i].

    Args:
        n: Length of the array.

    Returns:
        The last prefix sum value.
    """
    data = [0.0] * n
    for i in range(n):
        data[i] = ((i * 37 + 13) % 1000) / 100.0

    for i in range(1, n):
        data[i] = data[i - 1] + data[i]

    return data[n - 1]
