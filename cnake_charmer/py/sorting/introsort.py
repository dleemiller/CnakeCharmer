"""Introsort: quicksort with heapsort fallback at depth limit.

Keywords: sorting, introsort, quicksort, heapsort, hybrid, algorithm, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _sift_down(arr, start, end):
    """Sift down element at start in max-heap arr[start..end]."""
    root = start
    while True:
        child = 2 * root - start + 1
        if child > end:
            break
        if child + 1 <= end and arr[child] < arr[child + 1]:
            child += 1
        if arr[root] < arr[child]:
            arr[root], arr[child] = arr[child], arr[root]
            root = child
        else:
            break


def _heapsort(arr, lo, hi):
    """Heapsort arr[lo..hi] in place."""
    count = hi - lo + 1
    for start in range((count - 2) // 2 + lo, lo - 1, -1):
        _sift_down(arr, start, hi)
    end = hi
    while end > lo:
        arr[lo], arr[end] = arr[end], arr[lo]
        end -= 1
        _sift_down(arr, lo, end)


def _insertion_sort(arr, lo, hi):
    """Insertion sort arr[lo..hi]."""
    for i in range(lo + 1, hi + 1):
        temp = arr[i]
        j = i
        while j > lo and arr[j - 1] > temp:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = temp


def _introsort_impl(arr, lo, hi, depth_limit):
    """Iterative introsort implementation."""
    stack = [(lo, hi, depth_limit)]
    while stack:
        lo, hi, depth = stack.pop()
        size = hi - lo + 1
        if size <= 1:
            continue
        if size < 16:
            _insertion_sort(arr, lo, hi)
            continue
        if depth == 0:
            _heapsort(arr, lo, hi)
            continue

        # Partition with median-of-three
        mid = (lo + hi) // 2
        a, b, c = arr[lo], arr[mid], arr[hi]
        if a <= b:
            if b <= c:
                pivot_idx = mid
            elif a <= c:
                pivot_idx = hi
            else:
                pivot_idx = lo
        else:
            if a <= c:
                pivot_idx = lo
            elif b <= c:
                pivot_idx = hi
            else:
                pivot_idx = mid

        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        pivot = arr[hi]

        store = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[store], arr[j] = arr[j], arr[store]
                store += 1
        arr[store], arr[hi] = arr[hi], arr[store]

        stack.append((lo, store - 1, depth - 1))
        stack.append((store + 1, hi, depth - 1))


@python_benchmark(args=(200000,))
def introsort(n: int) -> list[int]:
    """Sort a deterministic array using introsort.

    Generates arr[i] = (i*31+17) % n, then sorts using introsort
    (quicksort with heapsort fallback at 2*log2(n) depth).

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = [(i * 31 + 17) % n for i in range(n)]
    depth_limit = 0
    tmp = n
    while tmp > 0:
        depth_limit += 1
        tmp >>= 1
    depth_limit *= 2
    _introsort_impl(arr, 0, n - 1, depth_limit)
    return arr
