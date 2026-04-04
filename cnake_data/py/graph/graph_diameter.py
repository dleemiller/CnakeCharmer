"""Find the diameter of a graph using repeated BFS.

Keywords: graph, diameter, bfs, longest shortest path, benchmark
"""

from collections import deque

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(4000,))
def graph_diameter(n: int) -> tuple:
    """Compute graph diameter via BFS from multiple sources.

    Builds a connected graph: ring i->(i+1)%n, plus edges i->(i*3+5)%n.
    Runs BFS from nodes 0, n//4, n//2, 3n//4 and takes the max eccentricity.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (diameter, sum of eccentricities from sampled sources,
                  total edges visited across all BFS runs).
    """
    if n < 2:
        return (0, 0, 0)

    # Build adjacency list: ring + cross-edges
    adj = [[] for _ in range(n)]
    for i in range(n):
        j = (i + 1) % n
        adj[i].append(j)
        adj[j].append(i)
    for i in range(n):
        k = (i * 3 + 5) % n
        if k != i:
            adj[i].append(k)
            adj[k].append(i)

    sources = [0, n // 4, n // 2, (3 * n) // 4]
    diameter = 0
    eccentricity_sum = 0
    total_edges_visited = 0

    for src in sources:
        dist = [-1] * n
        dist[src] = 0
        queue = deque()
        queue.append(src)
        max_dist = 0
        edges_visited = 0

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                edges_visited += 1
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    queue.append(v)

        eccentricity_sum += max_dist
        total_edges_visited += edges_visited
        if max_dist > diameter:
            diameter = max_dist

    return (diameter, eccentricity_sum, total_edges_visited)
