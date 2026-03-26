"""Conway's Game of Life simulation on an n x n grid for 50 generations.

Keywords: game of life, cellular automaton, simulation, grid, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(120,))
def game_of_life(n: int) -> int:
    """Run Conway's Game of Life on an n x n grid for 50 generations.

    Initial state: cell[i][j] = 1 if (i*7 + j*13 + 3) % 5 == 0 else 0.
    Returns the count of live cells after 50 steps.

    Args:
        n: Grid dimension (n x n).

    Returns:
        Tuple of (number of live cells after 50 generations, cell checksum).
    """
    generations = 50

    # Initialize grid as flat array
    size = n * n
    current = [0] * size
    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13 + 3) % 5 == 0:
                current[i * n + j] = 1

    nxt = [0] * size

    for _gen in range(generations):
        for i in range(n):
            for j in range(n):
                # Count neighbors
                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            neighbors += current[ni * n + nj]

                idx = i * n + j
                if current[idx] == 1:
                    nxt[idx] = 1 if (neighbors == 2 or neighbors == 3) else 0
                else:
                    nxt[idx] = 1 if neighbors == 3 else 0

        current, nxt = nxt, current

    total = 0
    cell_checksum = 0
    for i in range(size):
        total += current[i]
        cell_checksum += current[i] * ((i * 31 + 7) % 10000)
    return (total, cell_checksum)
