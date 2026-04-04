# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum all values in a binary tree built from deterministic data.

Keywords: algorithms, tree, binary tree, recursion, forward decl, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

cdef class TreeNode

cdef class TreeNode:
    """Binary tree node."""
    cdef int value
    cdef TreeNode left
    cdef TreeNode right

    def __cinit__(self, int value):
        self.value = value
        self.left = None
        self.right = None


cdef TreeNode _build_tree(int *values,
                          int lo, int hi):
    """Build a balanced binary tree."""
    if lo >= hi:
        return None
    cdef int mid = (lo + hi) // 2
    cdef TreeNode node = TreeNode(values[mid])
    node.left = _build_tree(values, lo, mid)
    node.right = _build_tree(values, mid + 1, hi)
    return node


cdef long long _tree_sum(TreeNode node):
    """Recursively sum all node values."""
    if node is None:
        return 0
    return (node.value
            + _tree_sum(node.left)
            + _tree_sum(node.right))


@cython_benchmark(syntax="cy", args=(50000,))
def forward_decl_tree_sum(int n):
    """Build balanced binary tree of n nodes, sum values."""
    cdef int *values
    cdef int i

    values = <int *>malloc(n * sizeof(int))
    if not values:
        raise MemoryError()

    for i in range(n):
        values[i] = (
            (<unsigned int>(<long long>i
             * <long long>2654435761)
             ^ <unsigned int>(<long long>i
             * <long long>1664525))
            & <unsigned int>0xFFFF
        )

    cdef TreeNode root = _build_tree(values, 0, n)
    cdef long long total = _tree_sum(root)
    free(values)
    return total
