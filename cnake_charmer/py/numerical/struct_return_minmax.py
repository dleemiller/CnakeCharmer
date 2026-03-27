"""Find min and max of an array using struct return pattern.

Demonstrates struct return value: helper returns both
min and max in a single struct.

Keywords: numerical, min, max, struct return, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def struct_return_minmax(n: int) -> float:
    """Find min and max of hash-derived array, return sum.

    Args:
        n: Number of elements.

    Returns:
        min_val + max_val of the array.
    """
    mask = 0xFFFFFFFF

    min_val = float("inf")
    max_val = float("-inf")

    for i in range(n):
        h = ((i * 2654435761) & mask) ^ ((i * 2246822519) & mask)
        val = (h & 0xFFFF) / 65535.0 * 200.0 - 100.0

        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

    return min_val + max_val
