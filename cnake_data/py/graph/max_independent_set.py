"""Greedy maximum independent set on a deterministic graph.

Keywords: graph, independent set, greedy, optimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def max_independent_set(n: int) -> int:
    """Compute greedy maximum independent set size.

    Builds a deterministic undirected graph with n nodes. Each node i has edges:
      i -- (i*3+1)%n
      i -- (i*7+2)%n

    Uses a neighbor-marking approach: when a node is added to the set,
    its two neighbors are marked as excluded.

    Greedy algorithm: iterate nodes in order, add to independent set if
    not excluded.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Size of the greedy independent set.
    """
    excluded = [False] * n
    count = 0

    for i in range(n):
        if not excluded[i]:
            count += 1
            j1 = (i * 3 + 1) % n
            j2 = (i * 7 + 2) % n
            excluded[j1] = True
            excluded[j2] = True

    return count
