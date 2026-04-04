"""
Insertion sort algorithm.

Keywords: sorting, insertion sort, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def insertion_sort(n: int) -> list[int]:
    """Sort a reversed list of n integers using insertion sort.

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = list(range(n, 0, -1))

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr
