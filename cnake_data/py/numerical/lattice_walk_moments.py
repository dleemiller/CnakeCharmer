"""Simulate a deterministic 2D lattice walk and summarize moments.

Keywords: numerical, random walk, lattice, checksum, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Walker2D:
    def __init__(self) -> None:
        self.x = 0
        self.y = 0

    def step(self, code: int) -> None:
        if code == 0:
            self.x += 1
        elif code == 1:
            self.x -= 1
        elif code == 2:
            self.y += 1
        else:
            self.y -= 1


@python_benchmark(args=(300000,))
def lattice_walk_moments(n: int) -> tuple:
    """Run an LCG-driven walk and return final and extremal statistics."""
    walker = Walker2D()
    state = 2463534242
    max_d2 = 0
    checksum = 0

    for i in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        step = state & 3
        walker.step(step)

        x, y = walker.x, walker.y
        d2 = x * x + y * y
        if d2 > max_d2:
            max_d2 = d2
        checksum = (checksum + ((x & 0xFFFF) << 16) + (y & 0xFFFF) + i) & 0xFFFFFFFF

    return (walker.x, walker.y, max_d2, checksum)
