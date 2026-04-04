"""Quicksort with median-of-three pivot selection.

Keywords: sorting, quicksort, median of three, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def quick_sort(n: int) -> list[int]:
    """Sort a deterministic array using quicksort with median-of-three pivot.

    Generates arr[i] = (i * 31 + 17) % n, then sorts using iterative
    quicksort with deterministic median-of-three pivot selection.

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = [(i * 31 + 17) % n for i in range(n)]
    _quicksort(arr, 0, n - 1)
    return arr


def _median_of_three(arr, lo, hi):
    mid = (lo + hi) // 2
    a, b, c = arr[lo], arr[mid], arr[hi]
    if a <= b:
        if b <= c:
            return mid
        elif a <= c:
            return hi
        else:
            return lo
    else:
        if a <= c:
            return lo
        elif b <= c:
            return hi
        else:
            return mid


def _quicksort(arr, lo, hi):
    # Iterative quicksort to avoid Python recursion limit
    stack = [(lo, hi)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

        # Use insertion sort for small partitions
        if hi - lo < 16:
            for i in range(lo + 1, hi + 1):
                temp = arr[i]
                j = i
                while j > lo and arr[j - 1] > temp:
                    arr[j] = arr[j - 1]
                    j -= 1
                arr[j] = temp
            continue

        pivot_idx = _median_of_three(arr, lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        pivot = arr[hi]

        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]

        stack.append((lo, i - 1))
        stack.append((i + 1, hi))
