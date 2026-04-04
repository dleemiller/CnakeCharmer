"""Abelian sandpile model on a 2D grid.

Keywords: simulation, sandpile, abelian, toppling, cellular automaton, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def sandpile(n: int) -> int:
    """Simulate abelian sandpile on an n x n grid.

    Places 4*n grains at the center cell, then topples until stable.
    A cell topples when it has >= 4 grains, sending 1 grain to each
    of its 4 neighbors. Grains falling off edges are lost.

    Args:
        n: Grid dimension (n x n).

    Returns:
        Count of cells with >= 1 grain in the final stable configuration.
    """
    grid = [[0] * n for _ in range(n)]
    center = n // 2
    grid[center][center] = 4 * n

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if grid[i][j] >= 4:
                    changed = True
                    spill = grid[i][j] // 4
                    grid[i][j] = grid[i][j] % 4
                    if i > 0:
                        grid[i - 1][j] += spill
                    if i < n - 1:
                        grid[i + 1][j] += spill
                    if j > 0:
                        grid[i][j - 1] += spill
                    if j < n - 1:
                        grid[i][j + 1] += spill

    count = 0
    for i in range(n):
        for j in range(n):
            if grid[i][j] >= 1:
                count += 1
    return count
