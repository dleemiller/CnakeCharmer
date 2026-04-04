# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Trapping rain water problem (Cython-optimized).

Keywords: leetcode, trapping rain water, two pointer, elevation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def trapping_rain_water(int n):
    """Compute trapped rain water using two pointers with C arrays."""
    cdef int *heights = <int *>malloc(n * sizeof(int))
    if not heights:
        raise MemoryError()

    cdef int i
    cdef unsigned int val
    for i in range(n):
        val = <unsigned int>(i * <unsigned int>2654435761U)
        heights[i] = val % 100 + 1

    cdef long long total_water = 0
    cdef int max_single = 0
    cdef int bars_with_water = 0
    cdef int left = 0
    cdef int right = n - 1
    cdef int left_max = 0
    cdef int right_max = 0
    cdef int water

    while left < right:
        if heights[left] < heights[right]:
            if heights[left] >= left_max:
                left_max = heights[left]
            else:
                water = left_max - heights[left]
                total_water += water
                if water > max_single:
                    max_single = water
                if water > 0:
                    bars_with_water += 1
            left += 1
        else:
            if heights[right] >= right_max:
                right_max = heights[right]
            else:
                water = right_max - heights[right]
                total_water += water
                if water > max_single:
                    max_single = water
                if water > 0:
                    bars_with_water += 1
            right -= 1

    free(heights)
    return (int(total_water), max_single, bars_with_water)
