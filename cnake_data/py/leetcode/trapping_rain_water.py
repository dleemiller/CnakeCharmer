"""Trapping rain water problem with detailed statistics.

Keywords: leetcode, trapping rain water, two pointer, elevation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def trapping_rain_water(n: int) -> tuple:
    """Compute trapped rain water using two-pointer technique.

    Generates heights h[i] = ((i * 2654435761) % 100) + 1,
    then uses two pointers to compute total trapped water.

    Args:
        n: Number of elevation bars.

    Returns:
        Tuple of (total_water, max_water_at_single_bar, num_bars_with_water).
    """
    heights = [0] * n
    for i in range(n):
        heights[i] = ((i * 2654435761) & 0xFFFFFFFF) % 100 + 1

    total_water = 0
    max_single = 0
    bars_with_water = 0

    left = 0
    right = n - 1
    left_max = 0
    right_max = 0

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

    return (total_water, max_single, bars_with_water)
