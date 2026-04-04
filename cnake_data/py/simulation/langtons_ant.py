"""Langton's ant simulation on an n x n grid.

Keywords: langton, ant, cellular automaton, simulation, grid, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def langtons_ant(n: int) -> int:
    """Simulate Langton's ant on an n x n grid for n*n steps.

    The ant starts at the center facing up. Rules:
    - On white cell: turn right 90, flip color, move forward.
    - On black cell: turn left 90, flip color, move forward.
    Grid wraps around (toroidal).

    Args:
        n: Grid dimension (n x n) and sqrt of step count.

    Returns:
        Count of black cells after n*n steps.
    """
    size = n * n
    grid = [0] * size  # 0=white, 1=black

    # Directions: 0=up, 1=right, 2=down, 3=left
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]

    x = n // 2
    y = n // 2
    direction = 0
    steps = size

    for _ in range(steps):
        idx = y * n + x
        if grid[idx] == 0:
            # White: turn right
            direction = (direction + 1) % 4
            grid[idx] = 1
        else:
            # Black: turn left
            direction = (direction + 3) % 4
            grid[idx] = 0
        x = (x + dx[direction]) % n
        y = (y + dy[direction]) % n

    count = 0
    for i in range(size):
        count += grid[i]
    return count
