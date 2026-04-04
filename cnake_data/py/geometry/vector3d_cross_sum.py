"""Compute cross products of 3D vector pairs and sum the results.

Keywords: vector, cross product, 3D, geometry, operator overloading, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Vec3D:
    """Simple 3D vector with operator overloading."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vec3D(-self.x, -self.y, -self.z)

    def cross(self, other):
        return Vec3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@python_benchmark(args=(100000,))
def vector3d_cross_sum(n: int) -> tuple:
    """Generate n 3D vector pairs, compute cross products, and sum.

    Uses operator overloading (__add__, __sub__, __neg__) to accumulate
    cross product results. Returns (x, y, z) of summed vector.

    Args:
        n: Number of vector pairs.

    Returns:
        Tuple of (x, y, z) sums.
    """
    total = Vec3D(0.0, 0.0, 0.0)

    for i in range(n):
        h1 = ((i * 2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((i * 1103515245 + 3) >> 8) & 0xFFFF
        h3 = ((i * 2246822519 + 5) >> 8) & 0xFFFF
        ax = (h1 % 201 - 100) / 10.0
        ay = (h2 % 201 - 100) / 10.0
        az = (h3 % 201 - 100) / 10.0

        h4 = ((i * 6364136223846793005 + 7) >> 16) & 0xFFFF
        h5 = ((i * 3935559000370003845 + 11) >> 16) & 0xFFFF
        h6 = ((i * 2862933555777941757 + 13) >> 16) & 0xFFFF
        bx = (h4 % 201 - 100) / 10.0
        by = (h5 % 201 - 100) / 10.0
        bz = (h6 % 201 - 100) / 10.0

        a = Vec3D(ax, ay, az)
        b = Vec3D(bx, by, bz)

        # Cross product using operator overloading for accumulation
        c = a.cross(b)
        # Use __neg__ and __sub__ to demonstrate operators:
        # total = total + c  is same as  total = total - (-c)
        neg_c = -c
        total = total - neg_c

    return (total.x, total.y, total.z)
