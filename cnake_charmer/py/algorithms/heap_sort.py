"""Heap sort on a deterministic integer array.

Keywords: algorithms, heap sort, sorting, in-place, sift-down, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def heap_sort(n: int) -> list:
    """Sort an array using heap sort with sift-down.

    Array: arr[i] = (i*31+17) % n.

    Args:
        n: Number of elements to sort.

    Returns:
        Sorted list of integers.
    """
    arr = [(i * 31 + 17) % n for i in range(n)]

    # Build max-heap (sift-down from last parent to root)
    for start in range(n // 2 - 1, -1, -1):
        # Sift down
        pos = start
        while True:
            child = 2 * pos + 1
            if child >= n:
                break
            if child + 1 < n and arr[child + 1] > arr[child]:
                child += 1
            if arr[child] > arr[pos]:
                arr[pos], arr[child] = arr[child], arr[pos]
                pos = child
            else:
                break

    # Extract elements from heap
    end = n - 1
    while end > 0:
        arr[0], arr[end] = arr[end], arr[0]
        # Sift down in reduced heap
        pos = 0
        while True:
            child = 2 * pos + 1
            if child >= end:
                break
            if child + 1 < end and arr[child + 1] > arr[child]:
                child += 1
            if arr[child] > arr[pos]:
                arr[pos], arr[child] = arr[child], arr[pos]
                pos = child
            else:
                break
        end -= 1

    return arr
