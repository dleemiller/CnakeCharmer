"""Evaluate expression trees built from literals and binary operations.

Keywords: algorithms, expression, tree, eval, inheritance, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Expr:
    """Base expression node."""

    def eval(self):
        return 0.0


class Literal(Expr):
    """Literal value expression."""

    def __init__(self, value):
        self.value = value

    def eval(self):
        return self.value


class BinOp(Expr):
    """Binary operation expression (add or multiply)."""

    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op  # 0 = add, 1 = multiply

    def eval(self):
        lv = self.left.eval()
        rv = self.right.eval()
        if self.op == 0:
            return lv + rv
        else:
            return lv * rv


def _build_tree(depth, seed):
    """Deterministically build an expression tree."""
    if depth <= 0:
        val = ((seed * 2654435761 + 17) % 10000) / 100.0 - 50.0
        return Literal(val), seed + 1
    op = ((seed * 1103515245 + 12345) >> 4) & 1
    left, seed = _build_tree(depth - 1, seed + 1)
    right, seed = _build_tree(depth - 1, seed + 1)
    return BinOp(left, right, op), seed


@python_benchmark(args=(50000,))
def expression_eval(n: int) -> float:
    """Build and evaluate n expression trees, sum results.

    Each tree has depth 4 (15 internal nodes + 16 leaves).

    Args:
        n: Number of expression trees to evaluate.

    Returns:
        Sum of all evaluation results.
    """
    total = 0.0
    for i in range(n):
        tree, _ = _build_tree(4, i * 37)
        total += tree.eval()

    return total
