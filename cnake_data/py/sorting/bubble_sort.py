"""
Bubble sort algorithm.

Keywords: sorting, bubble sort, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def bubble_sort(n: int) -> list[int]:
    """Sort a reversed list of n integers using bubble sort.

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = list(range(n, 0, -1))

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr
