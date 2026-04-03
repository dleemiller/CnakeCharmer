# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bubble sort with in-place swaps (Cython-optimized).

Keywords: bubble_sort, sorting, in_place, comparison, algorithm, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def bubble_sort_sum(int n):
    """Create a reversed list of n elements, bubble sort it, return checksum."""
    cdef list nums = list(range(n, 0, -1))
    cdef int i, j
    for i in range(n):
        for j in range(n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    cdef long total = 0
    for i in range(n):
        total += <int>nums[i]
    return total
