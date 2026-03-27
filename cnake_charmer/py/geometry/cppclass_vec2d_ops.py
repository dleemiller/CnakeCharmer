"""2D vector arithmetic with a simple Vec2d class.

Keywords: 2D vectors, dot product, vector sum, geometry, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


class Vec2d:
    """Simple 2D vector supporting addition and dot product."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: "Vec2d") -> "Vec2d":
        return Vec2d(self.x + other.x, self.y + other.y)

    def dot(self, other: "Vec2d") -> float:
        return self.x * other.x + self.y * other.y


@python_benchmark(args=(300000,))
def cppclass_vec2d_ops(n: int) -> tuple:
    """Generate n 2D vectors and compute aggregate statistics.

    Vector i: x_i = sin(i * 0.1), y_i = cos(i * 0.1).
    Accumulate all vectors to get a sum vector.
    Compute dot products of consecutive pairs (i, i+1) and sum them.

    Args:
        n: Number of vectors to generate.

    Returns:
        Tuple of (magnitude_squared_of_sum, sum_of_dot_products).
    """
    vecs = [Vec2d(math.sin(i * 0.1), math.cos(i * 0.1)) for i in range(n)]

    acc = Vec2d(0.0, 0.0)
    for v in vecs:
        acc = acc + v

    mag_sq = acc.x * acc.x + acc.y * acc.y

    dot_sum = 0.0
    for i in range(n - 1):
        dot_sum += vecs[i].dot(vecs[i + 1])

    return (mag_sq, dot_sum)
