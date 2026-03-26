"""Rule 110 one-dimensional cellular automaton.

Keywords: simulation, cellular automaton, rule 110, 1D, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def cellular_automaton_1d(n: int) -> int:
    """Run Rule 110 on n cells for 500 generations.

    Initial state: cell[i] = 1 if i == n//2 else 0.
    Rule 110 lookup: pattern -> new state.
    Returns count of live cells after 500 steps.

    Args:
        n: Number of cells.

    Returns:
        Count of live (1) cells after 500 generations.
    """
    generations = 500

    # Rule 110: binary 01101110
    rule = [0] * 8
    rule[1] = 1
    rule[2] = 1
    rule[3] = 1
    rule[4] = 0
    rule[5] = 1
    rule[6] = 1
    rule[7] = 0

    cells = [0] * n
    cells[n // 2] = 1
    new_cells = [0] * n

    for _ in range(generations):
        for i in range(n):
            left = cells[(i - 1) % n]
            center = cells[i]
            right = cells[(i + 1) % n]
            pattern = (left << 2) | (center << 1) | right
            new_cells[i] = rule[pattern]
        cells, new_cells = new_cells, cells

    count = 0
    for i in range(n):
        count += cells[i]
    return count
