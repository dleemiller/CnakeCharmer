"""Counting sort on a deterministic integer array.

Keywords: algorithms, counting sort, sorting, linear time, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def counting_sort(n: int) -> list:
    """Sort an array using counting sort for values in range [0, 999].

    Array: arr[i] = (i*31+17) % 1000.

    Args:
        n: Number of elements to sort.

    Returns:
        Sorted list of integers.
    """
    MAX_VAL = 1000
    arr = [(i * 31 + 17) % MAX_VAL for i in range(n)]

    # Count occurrences
    counts = [0] * MAX_VAL
    for val in arr:
        counts[val] += 1

    # Build sorted output
    result = []
    for val in range(MAX_VAL):
        for _ in range(counts[val]):
            result.append(val)

    return result
