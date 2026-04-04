"""Vector3 class operations with cross-product accumulation.

Keywords: geometry, class, vec3, cross product, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Vec3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z


@python_benchmark(args=(450000, 0.13))
def vec3_class_cross_norm(steps: int, bias: float) -> tuple:
    a = Vec3(1.0 + bias, 0.5 - bias, 0.25 + bias)
    b = Vec3(0.75, -0.25, 1.25)
    total = 0.0
    peak = 0.0
    for i in range(steps):
        c = a.cross(b)
        n2 = c.norm2()
        total += n2 + (i & 1) * 0.01
        if n2 > peak:
            peak = n2
        a = Vec3(c.x * 0.9 + 0.1, c.y * 0.8 - 0.05, c.z * 0.85 + 0.02)
    return (total, peak, a.norm2())
