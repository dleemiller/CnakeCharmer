"""Dirac delta approximation on a 2D signed distance field.

Keywords: level set, signed distance function, delta function, numerical, finite differences, gradient
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def level_set_delta(n: int, eps: float = 1.5) -> tuple:
    """Compute the smoothed Dirac delta on a 2D signed distance field.

    Builds an n×n radial signed distance field (distance from circle boundary),
    then applies the delta approximation using central-difference gradients:

        delta[x,y] = |∇φ| / (2ε) * (1 + cos(φ·π/ε))   where |φ| < ε

    Args:
        n:   Grid dimension.
        eps: Half-width of the delta support (default 1.5).

    Returns:
        Tuple of (total, max_val, count): sum of delta values, maximum delta
        value, and number of active (within-support) cells.
    """
    cx = n / 2.0
    cy = n / 2.0
    radius = n / 4.0
    pi = math.pi

    arr = [[math.sqrt((i - cx) ** 2 + (j - cy) ** 2) - radius for j in range(n)] for i in range(n)]

    total = 0.0
    max_val = 0.0
    count = 0

    for x in range(1, n - 1):
        for y in range(1, n - 1):
            phi = arr[x][y]
            if abs(phi) < eps:
                phi_x = (arr[x + 1][y] - arr[x - 1][y]) / 2.0
                phi_y = (arr[x][y + 1] - arr[x][y - 1]) / 2.0
                grad = math.sqrt(phi_x * phi_x + phi_y * phi_y)
                d = grad / (2.0 * eps) * (1.0 + math.cos(phi * pi / eps))
                total += d
                if d > max_val:
                    max_val = d
                count += 1

    return (total, max_val, count)
