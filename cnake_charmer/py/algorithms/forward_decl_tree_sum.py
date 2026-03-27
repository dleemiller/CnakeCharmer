"""Sum all values in a binary tree built from deterministic data.

Keywords: algorithms, tree, binary tree, recursion, forward declaration, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class TreeNode:
    """Binary tree node."""

    __slots__ = ("value", "left", "right")

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def _build_tree(values, lo, hi):
    """Build a balanced binary tree from values[lo:hi]."""
    if lo >= hi:
        return None
    mid = (lo + hi) // 2
    node = TreeNode(values[mid])
    node.left = _build_tree(values, lo, mid)
    node.right = _build_tree(values, mid + 1, hi)
    return node


def _tree_sum(node):
    """Recursively sum all node values."""
    if node is None:
        return 0
    return node.value + _tree_sum(node.left) + _tree_sum(node.right)


@python_benchmark(args=(50000,))
def forward_decl_tree_sum(n: int) -> int:
    """Build a balanced binary tree of n nodes and sum all values.

    Args:
        n: Number of nodes.

    Returns:
        Sum of all node values.
    """
    values = [0] * n
    for i in range(n):
        values[i] = ((i * 2654435761) ^ (i * 1664525)) & 0xFFFF
    root = _build_tree(values, 0, n)
    return _tree_sum(root)
