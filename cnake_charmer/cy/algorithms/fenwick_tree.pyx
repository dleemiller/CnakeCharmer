# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Build a Fenwick tree (Binary Indexed Tree) and compute all prefix sums (Cython-optimized).

Keywords: algorithms, fenwick tree, binary indexed tree, prefix sum, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def fenwick_tree(int n):
    """Build a Fenwick tree from n values and return the total of all prefix sums."""
    cdef long long *tree = <long long *>malloc((n + 1) * sizeof(long long))
    if tree == NULL:
        raise MemoryError("Failed to allocate tree")

    memset(tree, 0, (n + 1) * sizeof(long long))

    cdef int i, idx, val
    cdef long long s, total

    # Build Fenwick tree (1-indexed)
    for i in range(n):
        val = (i * 7 + 3) % 100
        idx = i + 1
        while idx <= n:
            tree[idx] += val
            idx += idx & (-idx)

    # Query all prefix sums and accumulate
    total = 0
    for i in range(n):
        idx = i + 1
        s = 0
        while idx > 0:
            s += tree[idx]
            idx -= idx & (-idx)
        total += s

    free(tree)
    return int(total)
