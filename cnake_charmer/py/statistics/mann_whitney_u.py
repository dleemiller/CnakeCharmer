"""Compute Mann-Whitney U statistic for two deterministic samples.

Keywords: statistics, mann-whitney, u-test, nonparametric, rank sum, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def mann_whitney_u(n: int) -> tuple:
    """Compute Mann-Whitney U statistic for two deterministic samples.

    Sample A: a[i] = (i*17+3) % 2003 for i in range(n).
    Sample B: b[i] = (i*31+7) % 1999 for i in range(n).
    Uses the direct O(n^2) counting method.

    Args:
        n: Size of each sample.

    Returns:
        Tuple of (u_stat, n_greater, tied_count).
    """
    a = [(i * 17 + 3) % 2003 for i in range(n)]
    b = [(i * 31 + 7) % 1999 for i in range(n)]

    n_greater = 0
    tied_count = 0
    for i in range(n):
        ai = a[i]
        for j in range(n):
            bj = b[j]
            if ai > bj:
                n_greater += 1
            elif ai == bj:
                tied_count += 1

    u_stat = n_greater + tied_count * 0.5

    return (u_stat, n_greater, tied_count)
