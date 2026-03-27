"""Diffusion-limited aggregation simulation with deterministic LCG.

Keywords: simulation, DLA, aggregation, diffusion, random walk, LCG, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def diffusion_limited_agg(n: int) -> tuple:
    """Simulate DLA with n particles on a grid using deterministic LCG random walk.

    Grid size 31x31, seed at center. Each particle starts from a deterministic
    position on the grid boundary and random-walks until it touches an occupied
    cell or exceeds max steps. Re-launches if it exits bounds.
    Uses LCG: state = (state * 1103515245 + 12345) & 0x7FFFFFFF.

    Args:
        n: Number of particles to release.

    Returns:
        Tuple of (occupied_count, max_radius_squared, center_density).
    """
    grid_size = 31
    half = grid_size // 2
    max_steps = 20000
    perim = 4 * (grid_size - 1)

    # Flat grid
    grid = [0] * (grid_size * grid_size)
    grid[half * grid_size + half] = 1
    occupied = 1

    lcg = 42
    max_r2 = 0

    for _ in range(n):
        # Pick a start on the perimeter
        lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
        edge_pos = (lcg >> 16) % perim
        if edge_pos < grid_size:
            px = edge_pos
            py_ = 0
        elif edge_pos < 2 * grid_size - 1:
            px = grid_size - 1
            py_ = edge_pos - grid_size + 1
        elif edge_pos < 3 * grid_size - 2:
            px = grid_size - 1 - (edge_pos - 2 * grid_size + 2)
            py_ = grid_size - 1
        else:
            px = 0
            py_ = grid_size - 1 - (edge_pos - 3 * grid_size + 3)

        for _step in range(max_steps):
            # Check 4 neighbors for occupied cell
            found = 0
            if px > 0 and grid[py_ * grid_size + px - 1] == 1:
                found = 1
            if found == 0 and px < grid_size - 1 and grid[py_ * grid_size + px + 1] == 1:
                found = 1
            if found == 0 and py_ > 0 and grid[(py_ - 1) * grid_size + px] == 1:
                found = 1
            if found == 0 and py_ < grid_size - 1 and grid[(py_ + 1) * grid_size + px] == 1:
                found = 1

            if found == 1:
                grid[py_ * grid_size + px] = 1
                occupied += 1
                r2 = (px - half) * (px - half) + (py_ - half) * (py_ - half)
                if r2 > max_r2:
                    max_r2 = r2
                break

            # Random walk step
            lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
            d = (lcg >> 16) % 4
            if d == 0:
                px += 1
            elif d == 1:
                py_ += 1
            elif d == 2:
                px -= 1
            else:
                py_ -= 1

            # Re-launch if out of bounds
            if px < 0 or px >= grid_size or py_ < 0 or py_ >= grid_size:
                lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
                edge_pos = (lcg >> 16) % perim
                if edge_pos < grid_size:
                    px = edge_pos
                    py_ = 0
                elif edge_pos < 2 * grid_size - 1:
                    px = grid_size - 1
                    py_ = edge_pos - grid_size + 1
                elif edge_pos < 3 * grid_size - 2:
                    px = grid_size - 1 - (edge_pos - 2 * grid_size + 2)
                    py_ = grid_size - 1
                else:
                    px = 0
                    py_ = grid_size - 1 - (edge_pos - 3 * grid_size + 3)

    # Center density: occupied cells in 5x5 around center
    center_density = 0
    for iy in range(half - 2, half + 3):
        for ix in range(half - 2, half + 3):
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                center_density += grid[iy * grid_size + ix]

    return (occupied, max_r2, center_density)
