# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Diffusion-limited aggregation simulation with deterministic LCG (Cython-optimized).

Keywords: simulation, DLA, aggregation, diffusion, random walk, LCG, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def diffusion_limited_agg(int n):
    """Simulate DLA with n particles on a grid using deterministic LCG random walk."""
    cdef int grid_size = 31
    cdef int half = grid_size // 2
    cdef int max_steps = 20000
    cdef int perim = 4 * (grid_size - 1)
    cdef int occupied = 1
    cdef long long lcg = 42
    cdef int max_r2 = 0
    cdef int i, d, px, py_, r2, _step, edge_pos
    cdef int ix, iy, center_density, found, stuck

    cdef int *grid = <int *>malloc(grid_size * grid_size * sizeof(int))
    if not grid:
        raise MemoryError()

    for i in range(grid_size * grid_size):
        grid[i] = 0
    grid[half * grid_size + half] = 1

    for i in range(n):
        lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
        edge_pos = <int>((lcg >> 16) % perim)
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

        stuck = 0
        for _step in range(max_steps):
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
                occupied = occupied + 1
                r2 = (px - half) * (px - half) + (py_ - half) * (py_ - half)
                if r2 > max_r2:
                    max_r2 = r2
                stuck = 1
                break

            lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
            d = <int>((lcg >> 16) % 4)
            if d == 0:
                px = px + 1
            elif d == 1:
                py_ = py_ + 1
            elif d == 2:
                px = px - 1
            else:
                py_ = py_ - 1

            if px < 0 or px >= grid_size or py_ < 0 or py_ >= grid_size:
                lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
                edge_pos = <int>((lcg >> 16) % perim)
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

    center_density = 0
    for iy in range(half - 2, half + 3):
        for ix in range(half - 2, half + 3):
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                center_density = center_density + grid[iy * grid_size + ix]

    free(grid)

    return (occupied, max_r2, center_density)
