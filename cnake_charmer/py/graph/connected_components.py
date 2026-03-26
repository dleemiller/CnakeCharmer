"""
Count connected components using union-find on a deterministic graph.

Keywords: graph, union-find, connected components, disjoint set, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def connected_components(n: int) -> int:
    """Count connected components using union-find with path compression.

    Creates n nodes and m=n*2 deterministic edges where edge i connects
    (i*7+3)%n to (i*13+7)%n. Uses union-find with path compression and
    union by rank.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (number of connected components, size of largest component).
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    m = n * 2
    for i in range(m):
        u = (i * 7 + 3) % n
        v = (i * 13 + 7) % n
        union(u, v)

    # Count components and find largest
    count = 0
    comp_size = {}
    for i in range(n):
        root = find(i)
        if root not in comp_size:
            comp_size[root] = 0
        comp_size[root] += 1
        if root == i:
            count += 1
    largest_size = max(comp_size.values())
    return (count, largest_size)
