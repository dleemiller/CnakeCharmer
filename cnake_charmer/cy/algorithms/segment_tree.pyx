# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Segment tree for range-max queries (Cython-optimized).

Keywords: algorithms, segment tree, range query, maximum, data structure, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef int *_tree
cdef int _n


cdef void _build(int node, int start, int end) noexcept nogil:
    cdef int mid, a, b
    if start == end:
        _tree[node] = (start * 31 + 17) % 1000
        return
    mid = (start + end) / 2
    _build(2 * node, start, mid)
    _build(2 * node + 1, mid + 1, end)
    a = _tree[2 * node]
    b = _tree[2 * node + 1]
    _tree[node] = a if a > b else b


cdef int _query(int node, int start, int end, int qr) noexcept nogil:
    cdef int mid, a, b
    if end <= qr:
        return _tree[node]
    mid = (start + end) / 2
    if qr <= mid:
        return _query(2 * node, start, mid, qr)
    a = _tree[2 * node]
    b = _query(2 * node + 1, mid + 1, end, qr)
    return a if a > b else b


@cython_benchmark(syntax="cy", args=(100000,))
def segment_tree(int n):
    """Build a segment tree for range-max queries and sum all query results."""
    global _tree, _n
    cdef int size = 4 * n
    _tree = <int *>malloc(size * sizeof(int))
    if _tree == NULL:
        raise MemoryError("Failed to allocate segment tree")

    _n = n
    cdef int i, n_half, query_at_n_half, q
    cdef long long total = 0

    _build(1, 0, n - 1)

    n_half = n / 2
    query_at_n_half = 0
    for i in range(n):
        q = _query(1, 0, n - 1, i)
        total += q
        if i == n_half:
            query_at_n_half = q

    free(_tree)
    return (int(total), query_at_n_half)
