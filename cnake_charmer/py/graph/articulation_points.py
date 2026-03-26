"""Count articulation points in a deterministic graph via DFS.

Keywords: graph, articulation points, cut vertices, dfs, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def articulation_points(n: int) -> int:
    """Count articulation points in a graph using Tarjan's algorithm.

    Edges: tree i -> (i-1)//2 for i>0, plus cross-edge i -> (i*3+7)%n for i%5==0.

    Args:
        n: Number of nodes.

    Returns:
        Number of articulation points.
    """
    if n < 2:
        return 0

    # Build adjacency list: binary tree + sparse cross-edges
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        parent = (i - 1) // 2
        adj[i].append(parent)
        adj[parent].append(i)
    # Sparse cross-edges every 5th node
    for i in range(0, n, 5):
        k = (i * 3 + 7) % n
        if k != i:
            adj[i].append(k)
            adj[k].append(i)

    disc = [-1] * n
    low = [-1] * n
    is_ap = [False] * n
    timer = 0

    # Iterative DFS to avoid recursion limit
    for start in range(n):
        if disc[start] != -1:
            continue

        # Stack: (node, parent, adj_index)
        stack = [(start, -1, 0)]
        disc[start] = timer
        low[start] = timer
        timer += 1
        child_count = [0] * n

        while stack:
            node, parent, idx = stack[-1]

            if idx < len(adj[node]):
                stack[-1] = (node, parent, idx + 1)
                neighbor = adj[node][idx]
                if disc[neighbor] == -1:
                    disc[neighbor] = timer
                    low[neighbor] = timer
                    timer += 1
                    child_count[node] += 1
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
                    # Check articulation point condition
                    if stack[-1][1] == -1:
                        # Root: AP if >1 child
                        pass
                    else:
                        if low[node] >= disc[parent_node]:
                            is_ap[parent_node] = True

        # Check root
        if child_count[start] > 1:
            is_ap[start] = True

    return sum(1 for x in is_ap if x)
