"""
Segment tree for range-max queries.

Keywords: algorithms, segment tree, range query, maximum, data structure, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def segment_tree(n: int) -> int:
    """Build a segment tree for range-max queries and sum all query results.

    Values: v[i] = (i*31+17) % 1000 for i in 0..n-1.
    Build segment tree, then query max over [0, i] for each i in 0..n-1.
    Return sum of all query results.

    Args:
        n: Number of values.

    Returns:
        Sum of range-max query results for all prefixes.
    """
    size = 4 * n
    tree = [0] * size

    def build(node, start, end):
        if start == end:
            tree[node] = (start * 31 + 17) % 1000
            return
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        a = tree[2 * node]
        b = tree[2 * node + 1]
        tree[node] = a if a > b else b

    def query(node, start, end, qr):
        if end <= qr:
            return tree[node]
        mid = (start + end) // 2
        if qr <= mid:
            return query(2 * node, start, mid, qr)
        a = tree[2 * node]
        b = query(2 * node + 1, mid + 1, end, qr)
        return a if a > b else b

    build(1, 0, n - 1)

    total = 0
    for i in range(n):
        total += query(1, 0, n - 1, i)

    return total
