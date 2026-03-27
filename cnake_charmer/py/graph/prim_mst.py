"""Prim's minimum spanning tree on a weighted graph.

Keywords: graph, prim, minimum spanning tree, mst, greedy, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def prim_mst(n: int) -> tuple:
    """Compute MST of n-node weighted graph using Prim's algorithm.

    Builds a deterministic complete-ish graph where each node i connects
    to a few neighbors with deterministic weights, then finds MST using
    Prim's with a simple linear scan (no priority queue).

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (total_weight, max_edge, min_edge) of the MST.
    """
    INF = 10**18

    # Build adjacency: for each node, edges to ~5 neighbors
    # Store as flat adjacency list: adj_target, adj_weight arrays
    # For simplicity, use adjacency matrix approach with key array
    # Each node i has edges to (i+1)%n, (i+3)%n, (i*7+2)%n, (i*13+5)%n, (i*31+11)%n
    # with weights based on both endpoints

    # Build sparse adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        targets = [
            (i + 1) % n,
            (i + 3) % n,
            (i * 7 + 2) % n,
            (i * 13 + 5) % n,
            (i * 31 + 11) % n,
        ]
        for t in targets:
            if t != i:
                w = ((i * 17 + t * 31 + 7) % 997) + 1
                adj[i].append((t, w))
                adj[t].append((i, w))

    # Prim's algorithm with linear scan
    in_mst = [False] * n
    key = [INF] * n
    key[0] = 0
    edge_from = [-1] * n

    total_weight = 0
    max_edge = 0
    min_edge = INF
    edges_added = 0

    for _ in range(n):
        # Find minimum key vertex not in MST
        u = -1
        min_key = INF
        for v in range(n):
            if not in_mst[v] and key[v] < min_key:
                min_key = key[v]
                u = v

        if u == -1:
            break

        in_mst[u] = True

        if edge_from[u] != -1:
            total_weight += key[u]
            edges_added += 1
            if key[u] > max_edge:
                max_edge = key[u]
            if key[u] < min_edge:
                min_edge = key[u]

        # Update keys of adjacent vertices
        for v, w in adj[u]:
            if not in_mst[v] and w < key[v]:
                key[v] = w
                edge_from[v] = u

    if min_edge == INF:
        min_edge = 0

    return (total_weight, max_edge, min_edge)
