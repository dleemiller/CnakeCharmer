"""Kruskal's minimum spanning tree.

Keywords: graph, MST, Kruskal, union-find, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def minimum_spanning_tree(n: int) -> int:
    """Compute MST weight using Kruskal's algorithm on n nodes.

    Edges: for each node i, j in 1..3: edge (i, (i*7+j)%n, weight=(i*j+3)%100).
    Total m = 3*n edges. Uses union-find with path compression and union by rank.
    Returns total MST weight.

    Args:
        n: Number of nodes.

    Returns:
        Total weight of the minimum spanning tree.
    """
    # Build edges
    edges = []
    for i in range(n):
        for j in range(1, 4):
            v = (i * 7 + j) % n
            w = (i * j + 3) % 100
            edges.append((w, i, v))

    edges.sort()

    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total_weight = 0
    edges_added = 0

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
            edges_added += 1
            if edges_added == n - 1:
                break

    return total_weight
