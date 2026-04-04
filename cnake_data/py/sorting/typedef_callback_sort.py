"""Sort array using callback-based comparison functions.

Keywords: sorting, callback, function pointer, typedef, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _compare_asc(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


def _compare_desc(a, b):
    if a > b:
        return -1
    elif a < b:
        return 1
    return 0


def _insertion_sort(arr, n, compare_fn):
    """Insertion sort using a comparison callback."""
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and compare_fn(arr[j], key) > 0:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


@python_benchmark(args=(50000,))
def typedef_callback_sort(n: int) -> int:
    """Sort array ascending then descending using callback comparators.

    Array values: arr[i] = (i * 1103515245 + 12345) % 1000000.
    Sort ascending, take checksum, sort descending, add checksum.

    Args:
        n: Number of elements.

    Returns:
        Checksum: sum of first 10 elements after ascending + first 10 after descending.
    """
    arr = [0] * n
    for i in range(n):
        arr[i] = ((i * 1103515245 + 12345) & 0x7FFFFFFF) % 1000000

    # Ascending sort
    arr.sort()
    checksum = 0
    for i in range(min(10, n)):
        checksum += arr[i]

    # Descending sort
    arr.sort(reverse=True)
    for i in range(min(10, n)):
        checksum += arr[i]

    return checksum
