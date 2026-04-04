"""Check if a generated graph is bipartite using BFS coloring.

Keywords: bipartite, graph, bfs, coloring, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def bipartite_check(n: int) -> int:
    """Check bipartiteness of an n-node graph using BFS coloring.

    Edges: i -> (i*3+1) % n, i -> (i*7+2) % n for each node i.
    Returns (is_bipartite << 20) | colored_count where is_bipartite is
    1 if the graph is bipartite, 0 otherwise, and colored_count is the
    number of nodes that were successfully colored.

    Args:
        n: Number of nodes.

    Returns:
        Encoded result: (is_bipartite << 20) | colored_count.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        u = (i * 3 + 1) % n
        v = (i * 7 + 2) % n
        adj[i].append(u)
        adj[u].append(i)
        adj[i].append(v)
        adj[v].append(i)

    color = [-1] * n
    is_bipartite = 1
    colored_count = 0

    for start in range(n):
        if color[start] != -1:
            continue
        # BFS
        queue = [start]
        color[start] = 0
        colored_count += 1
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in adj[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    colored_count += 1
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    is_bipartite = 0

    return (is_bipartite << 20) | colored_count
