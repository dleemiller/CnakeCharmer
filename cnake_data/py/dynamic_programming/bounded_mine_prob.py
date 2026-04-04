"""Count bounded integer compositions and aggregate occupancy probabilities.

Keywords: dynamic programming, bounded compositions, probability, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(36, 54, 3, 24))
def bounded_mine_prob(num_cells: int, total_mines: int, max_per_cell: int, repeats: int) -> tuple:
    """Compute bounded composition counts and occupancy probabilities."""

    def count_ways(cells: int, mines: int, cap: int) -> float:
        dp = [0.0] * (mines + 1)
        dp[0] = 1.0
        for _ in range(cells):
            nxt = [0.0] * (mines + 1)
            for used in range(mines + 1):
                if dp[used] == 0.0:
                    continue
                max_add = cap
                if used + max_add > mines:
                    max_add = mines - used
                for add in range(max_add + 1):
                    nxt[used + add] += dp[used]
            dp = nxt
        return dp[mines]

    ways_sum = 0.0
    prob_sum = 0.0
    last_prob = 0.0

    for r in range(repeats):
        mines = total_mines - (r % (max_per_cell + 1))
        if mines < 0:
            mines = 0
        total = count_ways(num_cells, mines, max_per_cell)
        without_first = count_ways(num_cells - 1, mines, max_per_cell)
        prob = 0.0 if total == 0.0 else 1.0 - without_first / total
        ways_sum += total
        prob_sum += prob
        last_prob = prob

    return (ways_sum, prob_sum, last_prob)
