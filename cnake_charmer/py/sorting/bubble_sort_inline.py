"""Bubble sort with in-place swaps.

Demonstrates the classic O(n^2) bubble sort algorithm with typed
loop variables for Cython speedup.

Keywords: bubble_sort, sorting, in_place, comparison, algorithm
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def bubble_sort_sum(n: int) -> int:
    """Create a reversed list of n elements, bubble sort it, return checksum.

    Args:
        n: Size of the list to sort.

    Returns:
        Sum of sorted list elements (for verification).
    """
    nums = list(range(n, 0, -1))
    for i in range(len(nums)):
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    total = 0
    for v in nums:
        total += v
    return total
