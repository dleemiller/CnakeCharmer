"""
Merge sort algorithm.

Keywords: sorting, merge sort, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def merge_sort(n: int) -> list[int]:
    """Sort a reversed list of n integers using merge sort.

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = list(range(n, 0, -1))
    _merge_sort(arr, 0, n - 1)
    return arr


def _merge_sort(arr: list[int], left: int, right: int) -> None:
    if left < right:
        mid = (left + right) // 2
        _merge_sort(arr, left, mid)
        _merge_sort(arr, mid + 1, right)
        _merge(arr, left, mid, right)


def _merge(arr: list[int], left: int, mid: int, right: int) -> None:
    left_half = arr[left : mid + 1]
    right_half = arr[mid + 1 : right + 1]

    i = 0
    j = 0
    k = left

    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1

    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1
