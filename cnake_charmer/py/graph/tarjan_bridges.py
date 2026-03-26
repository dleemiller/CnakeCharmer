"""Find bridges in an undirected graph using Tarjan's bridge-finding algorithm.

Keywords: graph, bridges, tarjan, dfs, cut edges, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def tarjan_bridges(n: int) -> tuple:
    """Find bridges in a deterministic undirected graph.

    Edges: binary tree i->(i-1)//2 for i>0, plus cross-edges i->(i*7+3)%n for i%6==0.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (bridge count, sum of bridge endpoint pairs min(u,v)).
    """
    if n < 2:
        return (0, 0)

    # Build adjacency list: binary tree + sparse cross-edges
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        parent = (i - 1) // 2
        adj[i].append(parent)
        adj[parent].append(i)
    # Sparse cross-edges every 6th node
    for i in range(0, n, 6):
        k = (i * 7 + 3) % n
        if k != i:
            adj[i].append(k)
            adj[k].append(i)

    disc = [-1] * n
    low = [-1] * n
    timer = 0
    bridge_count = 0
    bridge_min_sum = 0

    # Iterative DFS
    for start in range(n):
        if disc[start] != -1:
            continue

        stack = [(start, -1, 0)]
        disc[start] = timer
        low[start] = timer
        timer += 1

        while stack:
            node, parent, idx = stack[-1]

            if idx < len(adj[node]):
                stack[-1] = (node, parent, idx + 1)
                neighbor = adj[node][idx]
                if disc[neighbor] == -1:
                    disc[neighbor] = timer
                    low[neighbor] = timer
                    timer += 1
                    stack.append((neighbor, node, 0))
                elif neighbor != parent:
                    if disc[neighbor] < low[node]:
                        low[node] = disc[neighbor]
            else:
                stack.pop()
                if stack:
                    parent_node = stack[-1][0]
                    if low[node] < low[parent_node]:
                        low[parent_node] = low[node]
                    # Bridge condition: low[node] > disc[parent_node]
                    if low[node] > disc[parent_node]:
                        bridge_count += 1
                        bridge_min_sum += min(node, parent_node)

    return (bridge_count, bridge_min_sum)
