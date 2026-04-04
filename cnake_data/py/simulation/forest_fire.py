"""Forest fire simulation on a 2D grid.

Keywords: simulation, forest fire, cellular automaton, 2D grid, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def forest_fire(n: int) -> int:
    """Simulate a forest fire on an n x n grid for 50 steps.

    Cell states: 0=empty, 1=tree, 2=fire.
    Rules each step:
      - Fire -> empty
      - Tree -> fire if any neighbor is on fire
      - Empty -> tree if (i*7+j*13+step*31)%100 < 1 (deterministic growth)
    Initial: tree if (i*17+j*11)%3 != 0, else empty.
    Ignition: cell (0,0) starts on fire.

    Args:
        n: Grid dimension (n x n).

    Returns:
        Count of trees in the final grid.
    """
    steps = 50

    # Initialize grid
    grid = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i * 17 + j * 11) % 3 != 0:
                grid[i][j] = 1

    grid[0][0] = 2  # Start fire

    for step in range(steps):
        new_grid = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 2:
                    new_grid[i][j] = 0  # Fire burns out
                elif grid[i][j] == 1:
                    # Check neighbors for fire
                    on_fire = False
                    if i > 0 and grid[i - 1][j] == 2:
                        on_fire = True
                    if i < n - 1 and grid[i + 1][j] == 2:
                        on_fire = True
                    if j > 0 and grid[i][j - 1] == 2:
                        on_fire = True
                    if j < n - 1 and grid[i][j + 1] == 2:
                        on_fire = True
                    if on_fire:
                        new_grid[i][j] = 2
                    else:
                        new_grid[i][j] = 1
                else:
                    # Empty: deterministic growth
                    if (i * 7 + j * 13 + step * 31) % 100 < 1:
                        new_grid[i][j] = 1
                    else:
                        new_grid[i][j] = 0
        grid = new_grid

    tree_count = 0
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                tree_count += 1
    return tree_count
