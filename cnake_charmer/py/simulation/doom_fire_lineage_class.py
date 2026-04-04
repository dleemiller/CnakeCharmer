"""Class-based doom-fire style heat propagation simulation.

Keywords: simulation, class, cellular automata, heat map, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class DoomFire:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.heat = [0] * (width * height)

    def ignite_bottom(self, base: int) -> None:
        off = (self.height - 1) * self.width
        for x in range(self.width):
            self.heat[off + x] = (base + x * 3) & 255

    def spread_step(self, cooling: int, jitter: int) -> None:
        w = self.width
        h = self.height
        src = self.heat
        dst = src[:]
        for y in range(1, h):
            row = y * w
            up = (y - 1) * w
            for x in range(w):
                decay = cooling + ((x + y + jitter) & 3)
                v = src[row + x] - decay
                if v < 0:
                    v = 0
                nx = x - (jitter & 1)
                if nx < 0:
                    nx += w
                dst[up + nx] = v
        self.heat = dst


@python_benchmark(args=(90, 70, 140, 3, 37))
def doom_fire_lineage_class(width: int, height: int, steps: int, cooling: int, seed: int) -> tuple:
    fire = DoomFire(width, height)
    fire.ignite_bottom((seed * 29) & 255)

    checksum = 0
    for t in range(steps):
        fire.spread_step(cooling, seed + t)
        if (t & 15) == 0:
            checksum = (checksum + fire.heat[(t * 7) % (width * height)]) & 0xFFFFFFFF

    total = 0
    nonzero = 0
    peak = 0
    for v in fire.heat:
        total += v
        if v > 0:
            nonzero += 1
        if v > peak:
            peak = v

    return (total, nonzero, peak, checksum)
