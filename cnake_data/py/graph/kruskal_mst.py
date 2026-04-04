"""Kruskal's MST using union-find on a deterministic weighted edge set.

Keywords: graph, MST, Kruskal, union-find, spanning tree, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80000,))
def kruskal_mst(n: int) -> tuple:
    """Compute MST using Kruskal's algorithm with union-find.

    Edges: for each node i, edges to (i*5+1)%n with weight ((i*3+2)%97)+1,
    and to (i*11+7)%n with weight ((i*7+5)%89)+1.
    Total m = 2*n edges.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (total MST weight, number of MST edges, min edge weight in MST).
    """
    # Build edge list
    edges = [None] * (2 * n)
    for i in range(n):
        w1 = ((i * 3 + 2) % 97) + 1
        v1 = (i * 5 + 1) % n
        edges[2 * i] = (w1, i, v1)
        w2 = ((i * 7 + 5) % 89) + 1
        v2 = (i * 11 + 7) % n
        edges[2 * i + 1] = (w2, i, v2)

    edges.sort()

    # Union-Find with path compression and union by rank
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total_weight = 0
    edge_count = 0
    min_edge_weight = 0

    for w, u, v in edges:
        ru = find(u)
        rv = find(v)
        if ru != rv:
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            total_weight += w
            edge_count += 1
            if edge_count == 1:
                min_edge_weight = w
            if edge_count == n - 1:
                break

    return (total_weight, edge_count, min_edge_weight)
