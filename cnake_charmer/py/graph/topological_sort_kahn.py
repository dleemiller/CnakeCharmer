"""Kahn's algorithm for topological sort on a DAG.

Keywords: graph, topological sort, kahn, dag, bfs, ordering, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def topological_sort_kahn(n: int) -> tuple:
    """Topological sort of n-node DAG using Kahn's algorithm.

    Builds a deterministic DAG where edges go from lower to higher numbered
    nodes, then computes topological ordering.

    Args:
        n: Number of nodes in the DAG.

    Returns:
        Tuple of (order_first, order_last, order_mid) from the topological ordering.
    """
    # Build DAG: edges from i to j where j > i
    # Each node i has edges to deterministic targets > i
    adj = [[] for _ in range(n)]
    in_degree = [0] * n

    for i in range(n):
        t1 = i + 1 + (i * 7 + 3) % min(20, n - i) if i + 1 < n else -1
        t2 = i + 1 + (i * 13 + 7) % min(15, n - i) if i + 1 < n else -1
        t3 = i + 1 + (i * 31 + 11) % min(10, n - i) if i + 1 < n else -1

        seen = set()
        for t in (t1, t2, t3):
            if t is not None and 0 <= t < n and t != i and t not in seen:
                adj[i].append(t)
                in_degree[t] += 1
                seen.add(t)

    # Kahn's algorithm
    queue = [0] * n
    head = 0
    tail = 0

    for i in range(n):
        if in_degree[i] == 0:
            queue[tail] = i
            tail += 1

    order = [0] * n
    idx = 0

    while head < tail:
        u = queue[head]
        head += 1
        order[idx] = u
        idx += 1

        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue[tail] = v
                tail += 1

    mid = idx // 2
    return (order[0], order[idx - 1], order[mid])
