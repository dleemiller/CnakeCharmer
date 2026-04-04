"""DFS-based topological sort on a deterministic DAG.

Keywords: graph, topological sort, DFS, DAG, ordering, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def topological_sort_dfs(n: int) -> int:
    """Compute sum of topological order positions using DFS-based topo sort.

    Builds a DAG with n nodes. Edge i -> j exists if j = (i*3+1)%n and j > i.
    Uses iterative DFS to produce a topological ordering, then returns
    the sum of position indices.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Sum of topological order positions (position of node i in topo order).
    """
    # Build adjacency list (DAG: only edges where j > i)
    adj = [[] for _ in range(n)]
    for i in range(n):
        j = (i * 3 + 1) % n
        if j > i:
            adj[i].append(j)

    # Iterative DFS-based topological sort
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    order = []

    for start in range(n):
        if color[start] != WHITE:
            continue
        stack = [(start, 0)]
        color[start] = GRAY
        while stack:
            node, idx = stack.pop()
            if idx < len(adj[node]):
                stack.append((node, idx + 1))
                nb = adj[node][idx]
                if color[nb] == WHITE:
                    color[nb] = GRAY
                    stack.append((nb, 0))
            else:
                color[node] = BLACK
                order.append(node)

    order.reverse()

    # Sum of positions
    total = 0
    for pos in range(n):
        total += pos
    return total
