"""Affine updates through a lightweight class object.

Keywords: algorithms, class, affine transform, iterative update, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Eggs:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def advance(self, x: float, y: float) -> tuple[float, float]:
        nx = self.a * x + self.b * y + 0.1
        ny = self.b * x - self.a * y + 0.05
        return nx, ny


@python_benchmark(args=(0.83, 0.17, 750000, 0.2, -0.1))
def eggs_affine_iterate(a: float, b: float, steps: int, x0: float, y0: float) -> tuple:
    e = Eggs(a, b)
    x = x0
    y = y0
    checksum = 0.0
    for i in range(steps):
        x, y = e.advance(x, y)
        checksum += x * 0.7 + y * 0.3 + (i & 1) * 0.01
    return (x, y, checksum)
