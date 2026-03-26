"""
Minimum trials to find critical floor with k eggs and n floors.

Keywords: dynamic programming, egg drop, optimization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def egg_drop(n: int) -> int:
    """Compute minimum trials for egg drop problem using full DP table.

    Build the full DP table dp[e][f] = minimum trials with e eggs and f floors,
    for e in [1, eggs] and f in [0, n]. Uses the recurrence:
    dp[e][f] = 1 + min over x in [1,f] of max(dp[e-1][x-1], dp[e][f-x])

    Optimized using the monotonicity property: optimal x increases with f.

    Uses eggs=3. Returns dp[3][n].

    Args:
        n: Number of floors.

    Returns:
        Minimum number of trials in worst case with 3 eggs.
    """
    eggs = 3

    if n <= 0:
        return 0
    if n == 1:
        return 1

    # dp[e][f] for e in 0..eggs, f in 0..n
    # For e=1, dp[1][f] = f (linear search)
    # Build table using optimal drop floor monotonicity

    # Allocate as flat lists
    # prev_egg[f] = dp[e-1][f]
    # cur_egg[f] = dp[e][f]

    # Start with e=1: dp[1][f] = f
    prev_egg = list(range(n + 1))

    for _e in range(2, eggs + 1):
        cur_egg = [0] * (n + 1)
        opt_x = 1  # Optimal drop floor (monotone in f)
        for f in range(1, n + 1):
            # Find optimal x: minimize max(dp[e-1][x-1], dp[e][f-x])
            # Since dp[e-1][x-1] increases with x and dp[e][f-x] decreases,
            # we want the crossing point
            best = n + 1
            # Start search from previous optimal x
            for x in range(opt_x, f + 1):
                # breaks: dp[e-1][x-1], survives: cur_egg[f-x]
                val = 1 + max(prev_egg[x - 1], cur_egg[f - x])
                if val < best:
                    best = val
                    opt_x = x
                elif prev_egg[x - 1] > cur_egg[f - x]:
                    # Past the crossing point
                    break
            cur_egg[f] = best
        prev_egg = cur_egg

    return prev_egg[n]
