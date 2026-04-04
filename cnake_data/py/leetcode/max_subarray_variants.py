"""Find k=3 non-overlapping maximum subarrays using DP.

Keywords: leetcode, kadane, subarray, non-overlapping, dynamic programming, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def max_subarray_variants(n: int) -> tuple:
    """Find 3 non-overlapping subarrays of length w=n//10 with maximum total sum.

    Array is deterministic: a[i] = ((i * 73856093) ^ (i * 19349669)) % 201 - 100.
    Uses DP approach: compute prefix sums, then left-best and right-best arrays
    to find optimal middle subarray position.

    Args:
        n: Length of the array.

    Returns:
        Tuple of (max_sum, start_idx_of_first_subarray, subarray_count=3).
    """
    if n < 30:
        return (0, 0, 0)

    w = n // 10
    if w < 1:
        w = 1

    # Generate deterministic array
    a = [0] * n
    for i in range(n):
        a[i] = ((i * 73856093) ^ (i * 19349669)) % 201 - 100

    # Compute subarray sums of length w
    num_subs = n - w + 1
    sub_sum = [0] * num_subs
    s = 0
    for i in range(w):
        s += a[i]
    sub_sum[0] = s
    for i in range(1, num_subs):
        s = s + a[i + w - 1] - a[i - 1]
        sub_sum[i] = s

    # left_best[i] = index of best subarray starting at or before i
    left_best = [0] * num_subs
    left_best[0] = 0
    for i in range(1, num_subs):
        if sub_sum[i] > sub_sum[left_best[i - 1]]:
            left_best[i] = i
        else:
            left_best[i] = left_best[i - 1]

    # right_best[i] = index of best subarray starting at or after i
    right_best = [0] * num_subs
    right_best[num_subs - 1] = num_subs - 1
    for i in range(num_subs - 2, -1, -1):
        if sub_sum[i] >= sub_sum[right_best[i + 1]]:
            right_best[i] = i
        else:
            right_best[i] = right_best[i + 1]

    # Find best middle subarray
    max_sum = -999999999
    best_start = 0
    for mid in range(w, num_subs - w):
        total = sub_sum[left_best[mid - w]] + sub_sum[mid] + sub_sum[right_best[mid + w]]
        if total > max_sum:
            max_sum = total
            best_start = left_best[mid - w]

    return (max_sum, best_start, 3)
