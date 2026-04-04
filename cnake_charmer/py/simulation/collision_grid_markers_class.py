"""Class-based collision grid marker updates with block occupancy stats.

Keywords: simulation, class, grid, collision, occupancy, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class CollisionGrid:
    def __init__(self, width: int, height: int, block: int):
        self.width = width
        self.height = height
        self.block = block
        self.nx = width // block
        self.ny = height // block
        self.cells = [0] * (self.nx * self.ny)

    def _idx(self, x: int, y: int) -> int:
        return y * self.nx + x

    def move(self, old_x: int, old_y: int, new_x: int, new_y: int) -> None:
        oi = self._idx(old_x, old_y)
        ni = self._idx(new_x, new_y)
        self.cells[oi] -= 1
        self.cells[ni] += 1


@python_benchmark(args=(192, 192, 6, 2800, 120, 19))
def collision_grid_markers_class(
    width: int,
    height: int,
    block: int,
    n_particles: int,
    steps: int,
    seed: int,
) -> tuple:
    grid = CollisionGrid(width, height, block)
    nx = grid.nx
    ny = grid.ny

    px = [0] * n_particles
    py = [0] * n_particles
    vx = [0] * n_particles
    vy = [0] * n_particles

    for i in range(n_particles):
        x = (seed * 97 + i * 31) % nx
        y = (seed * 53 + i * 29) % ny
        px[i] = x
        py[i] = y
        vx[i] = 1 + ((seed + i) & 1)
        vy[i] = 1 + (((seed >> 1) + i) & 1)
        grid.cells[y * nx + x] += 1

    checksum = 0
    for t in range(steps):
        for i in range(n_particles):
            ox = px[i]
            oy = py[i]
            nx2 = (ox + vx[i]) % nx
            ny2 = (oy + vy[i]) % ny
            if ((i + t + seed) & 7) == 0:
                nx2 = (nx2 + 1) % nx
            if ((i + t + seed) & 11) == 0:
                ny2 = (ny2 + 1) % ny
            grid.move(ox, oy, nx2, ny2)
            px[i] = nx2
            py[i] = ny2
            checksum = (checksum + (nx2 + 3 * ny2 + grid.cells[ny2 * nx + nx2])) & 0xFFFFFFFF

    occupied = 0
    max_cell = 0
    for v in grid.cells:
        if v > 0:
            occupied += 1
        if v > max_cell:
            max_cell = v

    return (occupied, max_cell, checksum)
