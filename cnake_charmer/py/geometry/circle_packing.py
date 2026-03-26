"""Greedy circle packing density in a unit square with n candidate circles.

Keywords: geometry, circle packing, greedy, density, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def circle_packing(n: int) -> tuple:
    """Greedily pack circles into a unit square and return packing statistics.

    Candidate circles have deterministic centers and radii.  Each candidate is
    accepted only if it does not overlap any previously accepted circle and fits
    entirely within the unit square [0, 1] x [0, 1].

    Args:
        n: Number of candidate circles to try.

    Returns:
        Tuple of (total_area_covered, number_of_accepted_circles).
    """
    # Pre-generate candidates
    cx_list = [0.0] * n
    cy_list = [0.0] * n
    cr_list = [0.0] * n
    for i in range(n):
        # Deterministic pseudo-random positions and radii
        cx_list[i] = (i * 7 + 13) * 0.6180339887 % 1.0
        cy_list[i] = (i * 11 + 17) * 0.4142135624 % 1.0
        cr_list[i] = 0.005 + 0.04 * ((i * 3 + 7) % 50) / 50.0

    # Accepted circles stored as flat lists
    accepted_x = [0.0] * n
    accepted_y = [0.0] * n
    accepted_r = [0.0] * n
    count = 0
    total_area = 0.0

    for i in range(n):
        x = cx_list[i]
        y = cy_list[i]
        r = cr_list[i]

        # Check if circle fits inside unit square
        if x - r < 0.0 or x + r > 1.0 or y - r < 0.0 or y + r > 1.0:
            continue

        # Check overlap with all accepted circles
        overlap = False
        for j in range(count):
            dx = x - accepted_x[j]
            dy = y - accepted_y[j]
            min_dist = r + accepted_r[j]
            if dx * dx + dy * dy < min_dist * min_dist:
                overlap = True
                break

        if not overlap:
            accepted_x[count] = x
            accepted_y[count] = y
            accepted_r[count] = r
            count += 1
            total_area += math.pi * r * r

    return (total_area, count)
