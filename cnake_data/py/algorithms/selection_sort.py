"""
Selection sort algorithm.

Keywords: sorting, selection sort, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def selection_sort(n: int) -> list[int]:
    """Sort a reversed list of n integers using selection sort.

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = list(range(n, 0, -1))

    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr
