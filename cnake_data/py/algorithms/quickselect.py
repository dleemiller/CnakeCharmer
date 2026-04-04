"""Find kth smallest element using quickselect with deterministic pivoting.

Keywords: algorithms, quickselect, selection, partition, kth smallest, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def quickselect(n: int) -> tuple:
    """Find multiple kth-smallest elements using quickselect partition scheme.

    Builds array arr[i] = (i*37 + 13) % (n*2), then finds the elements at
    positions n//4, n//2, and 3*n//4 using the quickselect algorithm.

    Args:
        n: Number of elements in the array.

    Returns:
        Tuple of (element at n//4, element at n//2, element at 3*n//4).
    """
    arr = [(i * 37 + 13) % (n * 2) for i in range(n)]

    def _partition(a, lo, hi, pivot_idx):
        pivot_val = a[pivot_idx]
        a[pivot_idx], a[hi] = a[hi], a[pivot_idx]
        store = lo
        for i in range(lo, hi):
            if a[i] < pivot_val:
                a[store], a[i] = a[i], a[store]
                store += 1
        a[store], a[hi] = a[hi], a[store]
        return store

    def _quickselect(a, lo, hi, k):
        while lo < hi:
            # Median-of-three pivot
            mid = (lo + hi) // 2
            if a[lo] > a[mid]:
                a[lo], a[mid] = a[mid], a[lo]
            if a[lo] > a[hi]:
                a[lo], a[hi] = a[hi], a[lo]
            if a[mid] > a[hi]:
                a[mid], a[hi] = a[hi], a[mid]
            pivot_idx = _partition(a, lo, hi, mid)
            if pivot_idx == k:
                return a[k]
            elif pivot_idx < k:
                lo = pivot_idx + 1
            else:
                hi = pivot_idx - 1
        return a[lo]

    # Find three different order statistics on separate copies
    r1 = _quickselect(list(arr), 0, n - 1, n // 4)
    r2 = _quickselect(list(arr), 0, n - 1, n // 2)
    r3 = _quickselect(list(arr), 0, n - 1, (3 * n) // 4)

    return (r1, r2, r3)
