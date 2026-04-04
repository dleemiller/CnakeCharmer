"""Sliding window k-th largest using a sorted list with bisect.

Keywords: sliding window, k-th largest, sorted set, bisect, benchmark
"""

import bisect

from cnake_data.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(100000,))
def stl_set_kth_largest(n: int) -> tuple:
    """Sliding window k-th largest over n values.

    Values are generated as: val_i = (i * 2654435761) % 1_000_000_000
    Window size w=1000, k=100 (k-th largest = element at index w-k from the
    front of the sorted ascending window, i.e. index w-k).

    For each position i >= w-1, insert val_i, remove the oldest value
    val_{i-w}, and record the k-th largest element.

    Args:
        n: Total number of values to process.

    Returns:
        Tuple of (sum_of_kth_values % (10**9+7), final_kth_value).
    """
    W = 1000
    K = 100
    MOD_VAL = MOD

    vals = [(i * 2654435761) % 1_000_000_000 for i in range(n)]

    window: list[int] = []
    sum_kth = 0
    final_kth = 0

    for i in range(n):
        bisect.insort(window, vals[i])
        if i >= W:
            # remove oldest element vals[i - W]
            old = vals[i - W]
            pos = bisect.bisect_left(window, old)
            window.pop(pos)
        if i >= W - 1:
            # k-th largest: index from end is K-1, so index from front = len-K
            kth = window[len(window) - K]
            sum_kth = (sum_kth + kth) % MOD_VAL
            final_kth = kth

    return (sum_kth, final_kth)
