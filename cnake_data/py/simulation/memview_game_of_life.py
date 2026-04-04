"""
Conway's Game of Life on an n*n grid for 50 steps using flat lists, return live cell count.

Keywords: simulation, cellular automaton, game of life, 2D, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def memview_game_of_life(n: int) -> int:
    """Run Game of Life for 50 steps on n*n grid, return live cell count.

    Initial state: cell (i, j) is alive if ((i * 71 + j * 43 + 19) % 7) < 2.

    Args:
        n: Dimension of the square grid.

    Returns:
        Number of live cells after 50 steps.
    """
    steps = 50

    # Initialize grid
    grid = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            if ((i * 71 + j * 43 + 19) % 7) < 2:
                grid[i * n + j] = 1

    buf = [0] * (n * n)

    for _ in range(steps):
        for i in range(n):
            for j in range(n):
                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            neighbors += grid[ni * n + nj]
                alive = grid[i * n + j]
                if alive:
                    buf[i * n + j] = 1 if (neighbors == 2 or neighbors == 3) else 0
                else:
                    buf[i * n + j] = 1 if neighbors == 3 else 0

        # Swap
        grid, buf = buf, grid

    count = 0
    for i in range(n * n):
        count += grid[i]

    return count
