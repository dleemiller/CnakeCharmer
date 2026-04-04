"""Class-based particle motion with drag and path accumulation.

Keywords: simulation, class, drag, trajectory, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def advance(self, ax: float, ay: float, drag: float, dt: float) -> None:
        self.vx = (self.vx + ax * dt) * (1.0 - drag)
        self.vy = (self.vy + ay * dt) * (1.0 - drag)
        self.x += self.vx * dt
        self.y += self.vy * dt


@python_benchmark(args=(140, 1800, 0.008, 0.03))
def particle_drag_path_class(n: int, steps: int, dt: float, drag: float) -> tuple:
    ps = [Particle(i * 0.02, -i * 0.01, (i % 7) * 0.04, (i % 5) * -0.03) for i in range(n)]
    path_sum = 0.0
    for t in range(steps):
        for i, p in enumerate(ps):
            ax = ((t + i * 7) % 13 - 6) * 0.01
            ay = ((t + i * 5) % 11 - 5) * 0.01
            p.advance(ax, ay, drag, dt)
            path_sum += p.x * 0.3 + p.y * 0.7
    return (path_sum, ps[n // 2].x, ps[n // 2].y)
