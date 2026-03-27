"""Parallel element-wise addition of two arrays, return sum of result.

Keywords: numerical, vector, addition, parallel, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def prange_vector_add(n: int) -> float:
    """Add two deterministic arrays element-wise and return sum.

    Arrays are generated via hash-based formulas for reproducibility.

    Args:
        n: Length of each array.

    Returns:
        Sum of the element-wise addition result.
    """
    total = 0.0
    for i in range(n):
        a_i = (i * 2654435761 & 0xFFFFFFFF) / 4294967296.0
        b_i = (i * 2246822519 & 0xFFFFFFFF) / 4294967296.0
        total += a_i + b_i

    return total
