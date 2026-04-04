"""Pairwise particle momentum exchange using class objects.

Keywords: simulation, class, particle, momentum, interactions, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Particle:
    def __init__(self, mass: float, vx: float, vy: float):
        self.mass = mass
        self.vx = vx
        self.vy = vy

    def exchange(self, other: "Particle", k: float) -> None:
        dvx = (other.vx - self.vx) * k
        dvy = (other.vy - self.vy) * k
        self.vx += dvx / self.mass
        self.vy += dvy / self.mass
        other.vx -= dvx / other.mass
        other.vy -= dvy / other.mass


@python_benchmark(args=(90, 350, 0.07))
def particle_pair_momentum_class(n: int, rounds: int, coupling: float) -> tuple:
    ps = [Particle(1.0 + (i % 5) * 0.2, (i % 9) * 0.03, (i % 11) * -0.02) for i in range(n)]
    for r in range(rounds):
        for i in range(0, n - 1, 2):
            k = coupling + (r & 3) * 0.005
            ps[i].exchange(ps[i + 1], k)
    px = 0.0
    py = 0.0
    for p in ps:
        px += p.mass * p.vx
        py += p.mass * p.vy
    return (px, py, ps[n // 3].vx)
