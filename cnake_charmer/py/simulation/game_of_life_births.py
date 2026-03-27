"""Conway's Game of Life with birth tracking.

Keywords: simulation, game of life, cellular automaton, grid, births, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100,))
def game_of_life_births(n: int) -> tuple:
    """Run Game of Life for n generations on 80x80 grid, tracking births.

    Initial state seeded with LCG pseudo-random generator (seed=42).
    Approximately 40% of cells start alive.
    Counts total birth events (dead cell becomes alive) across all generations.

    Args:
        n: Number of generations to simulate.

    Returns:
        Tuple of (alive_count, alive_at_center, total_births).
    """
    rows = 80
    cols = 80
    size = rows * cols

    # Initialize grid with LCG pseudo-random (seed=42)
    current = [0] * size
    lcg = 42
    for i in range(size):
        lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
        if lcg % 100 < 40:
            current[i] = 1

    nxt = [0] * size
    total_births = 0

    for _gen in range(n):
        for i in range(rows):
            for j in range(cols):
                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbors += current[ni * cols + nj]

                idx = i * cols + j
                if current[idx] == 1:
                    nxt[idx] = 1 if (neighbors == 2 or neighbors == 3) else 0
                else:
                    if neighbors == 3:
                        nxt[idx] = 1
                        total_births += 1
                    else:
                        nxt[idx] = 0

        current, nxt = nxt, current

    alive_count = 0
    for i in range(size):
        alive_count += current[i]

    center = (rows // 2) * cols + (cols // 2)
    alive_at_center = current[center]

    return (alive_count, alive_at_center, total_births)
