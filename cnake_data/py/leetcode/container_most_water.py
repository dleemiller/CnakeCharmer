"""Container with most water (two pointer approach).

Keywords: leetcode, container most water, two pointer, greedy, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def container_most_water(n: int) -> tuple:
    """Find the container with most water using two pointers.

    Generates deterministic heights h[i] = ((i * 2654435761) % 1000000) + 1,
    then uses the two-pointer technique to find the maximum water area.

    Args:
        n: Number of vertical lines (heights).

    Returns:
        Tuple of (maximum area, left index of best container,
        right index of best container, number of pointer moves).
    """
    heights = [0] * n
    for i in range(n):
        heights[i] = ((i * 2654435761) & 0xFFFFFFFF) % 1000000 + 1

    max_area = 0
    best_left = 0
    best_right = 0
    moves = 0

    left = 0
    right = n - 1

    while left < right:
        h_left = heights[left]
        h_right = heights[right]
        area = h_left * (right - left) if h_left < h_right else h_right * (right - left)

        if area > max_area:
            max_area = area
            best_left = left
            best_right = right

        if h_left <= h_right:
            left += 1
        else:
            right -= 1
        moves += 1

    return (max_area, best_left, best_right, moves)
