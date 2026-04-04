"""Find min and max of hash-derived values and return their sum.

Keywords: ctuple, min, max, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def ctuple_minmax(n: int) -> float:
    """Compute min and max of n hash-derived values, return min + max.

    Args:
        n: Number of values to process.

    Returns:
        min_val + max_val as float.
    """
    h = (0 * 2654435761) & 0xFFFFFFFF
    first = (h / 4294967295.0) * 200.0 - 100.0
    mn = first
    mx = first
    for i in range(1, n):
        h = (i * 2654435761) & 0xFFFFFFFF
        val = (h / 4294967295.0) * 200.0 - 100.0
        if val < mn:
            mn = val
        if val > mx:
            mx = val
    return mn + mx
