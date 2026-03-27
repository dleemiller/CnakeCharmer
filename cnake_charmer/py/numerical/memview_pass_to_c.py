"""
Compute weighted sum by passing array data to a helper function (simulates C interop).

Keywords: numerical, weighted sum, interop, array, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _weighted_sum(data, n):
    """Compute weighted sum: sum(data[i] * (i + 1))."""
    result = 0.0
    for i in range(n):
        result += data[i] * (i + 1)
    return result


@python_benchmark(args=(100000,))
def memview_pass_to_c(n: int) -> float:
    """Create array, pass to helper function, return weighted sum.

    Data: data[i] = ((i * 67 + 23) % 300) / 15.0

    Args:
        n: Length of the array.

    Returns:
        Weighted sum as a float.
    """
    data = [0.0] * n
    for i in range(n):
        data[i] = ((i * 67 + 23) % 300) / 15.0

    return _weighted_sum(data, n)
