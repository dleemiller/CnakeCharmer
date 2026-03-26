"""Disjoint set union-find with path compression and union by rank.

Keywords: algorithms, union find, disjoint set, path compression, connected components, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def union_find(n: int) -> tuple:
    """Perform union-find operations on n elements with deterministic edges.

    Creates n elements, performs a series of union operations based on a
    deterministic edge pattern, then counts the number of connected components
    and the size of the largest component.

    Args:
        n: Number of elements.

    Returns:
        Tuple of (num_components, largest_component_size, total_unions_performed).
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
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    # Generate deterministic edges and perform unions
    unions_done = 0
    num_edges = 3 * n // 2
    for i in range(num_edges):
        a = (i * 31 + 7) % n
        b = (i * 67 + 13) % n
        if a != b and union(a, b):
            unions_done += 1

    # Count components and find largest
    comp_size = [0] * n
    num_components = 0
    for i in range(n):
        root = find(i)
        comp_size[root] += 1

    largest = 0
    for i in range(n):
        if comp_size[i] > 0:
            num_components += 1
            if comp_size[i] > largest:
                largest = comp_size[i]

    return (num_components, largest, unions_done)
