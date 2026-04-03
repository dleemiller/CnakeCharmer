def connected_components(n):
    """Count connected components in a sparse graph using union-find.

    Returns (num_components, size of largest component, number of union ops).
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    union_ops = 0

    for i in range(n):
        j = (i * 97 + 31) % n
        if i != j:
            ri = find(i)
            rj = find(j)
            if ri != rj:
                if rank[ri] < rank[rj]:
                    parent[ri] = rj
                elif rank[ri] > rank[rj]:
                    parent[rj] = ri
                else:
                    parent[rj] = ri
                    rank[ri] += 1
                union_ops += 1

        j = (i * 53 + 17) % n
        if i != j:
            ri = find(i)
            rj = find(j)
            if ri != rj:
                if rank[ri] < rank[rj]:
                    parent[ri] = rj
                elif rank[ri] > rank[rj]:
                    parent[rj] = ri
                else:
                    parent[rj] = ri
                    rank[ri] += 1
                union_ops += 1

    comp_sizes = {}
    for i in range(n):
        root = find(i)
        comp_sizes[root] = comp_sizes.get(root, 0) + 1

    num_components = len(comp_sizes)
    largest = max(comp_sizes.values()) if comp_sizes else 0

    return (num_components, largest, union_ops)
