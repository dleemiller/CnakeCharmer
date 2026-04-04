"""Merge overlapping intervals and return the count of merged intervals.

Keywords: leetcode, merge intervals, sorting, sweep, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def merge_intervals(n: int) -> int:
    """Generate n intervals, merge overlapping ones, return merged count.

    Interval i has start = i*3 % 1000, end = start + (i % 20 + 1).

    Args:
        n: Number of intervals.

    Returns:
        Number of merged intervals.
    """
    starts = [0] * n
    ends = [0] * n
    for i in range(n):
        s = (i * 3) % 1000
        starts[i] = s
        ends[i] = s + (i % 20 + 1)

    indices = sorted(range(n), key=lambda idx: (starts[idx], ends[idx]))

    merged_count = 0
    cur_end = ends[indices[0]]

    for k in range(1, n):
        idx = indices[k]
        s = starts[idx]
        e = ends[idx]
        if s <= cur_end:
            if e > cur_end:
                cur_end = e
        else:
            merged_count += 1
            cur_end = e
    merged_count += 1
    return merged_count
