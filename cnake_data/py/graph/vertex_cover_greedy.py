"""Greedy vertex cover approximation.

Keywords: graph, vertex cover, greedy, approximation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def vertex_cover_greedy(n: int) -> tuple:
    """Compute a greedy vertex cover on a deterministic graph.

    Edges: ring i->(i+1)%n, plus cross-edges i->(i*7+3)%n for i%3==0.
    Greedy strategy: pick the vertex with highest remaining degree,
    add to cover, remove its edges. Repeat until no edges remain.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (cover size, sum of covered vertex indices, total edges in graph).
    """
    if n < 2:
        return (0, 0, 0)

    # Build adjacency sets for easy edge removal tracking
    # Use degree array and edge list approach for efficiency
    # Build edge list
    edge_set = set()
    for i in range(n):
        j = (i + 1) % n
        u, v = min(i, j), max(i, j)
        edge_set.add((u, v))
    for i in range(0, n, 3):
        k = (i * 7 + 3) % n
        if k != i:
            u, v = min(i, k), max(i, k)
            edge_set.add((u, v))

    total_edges = len(edge_set)

    # Build neighbor lists from edge set
    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)

    degree = [len(adj[i]) for i in range(n)]
    in_cover = [False] * n
    removed = [False] * n
    cover_size = 0
    cover_index_sum = 0

    # Track active edges per node
    active_neighbors = [set(adj[i]) for i in range(n)]

    remaining_edges = total_edges

    while remaining_edges > 0:
        # Find node with max degree among non-removed
        best = -1
        best_deg = -1
        for i in range(n):
            if not removed[i] and degree[i] > best_deg:
                best_deg = degree[i]
                best = i

        if best == -1 or best_deg == 0:
            break

        # Add to cover
        in_cover[best] = True
        removed[best] = True
        cover_size += 1
        cover_index_sum += best

        # Remove all edges incident to best
        for nb in active_neighbors[best]:
            if not removed[nb]:
                degree[nb] -= 1
                active_neighbors[nb].discard(best)
                remaining_edges -= 1

        degree[best] = 0
        active_neighbors[best].clear()

    return (cover_size, cover_index_sum, total_edges)
