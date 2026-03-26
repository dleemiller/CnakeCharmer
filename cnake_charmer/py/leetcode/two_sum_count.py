"""Count pairs in a deterministic array that sum to a target value.

Keywords: leetcode, two sum, hash map, counting, pairs, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def two_sum_count(n: int) -> int:
    """Count pairs (i, j) with i < j where arr[i] + arr[j] == target.

    Array is generated as arr[i] = (i * 31 + 17) % n.
    Target is n // 2.

    Args:
        n: Size of the array.

    Returns:
        Number of pairs summing to target.
    """
    target = n // 2
    arr = [(i * 31 + 17) % n for i in range(n)]
    counts: dict[int, int] = {}
    total = 0
    for val in arr:
        complement = target - val
        if complement in counts:
            total += counts[complement]
        counts[val] = counts.get(val, 0) + 1
    return total
