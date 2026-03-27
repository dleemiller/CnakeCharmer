# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate expression trees using cdef class inheritance (Cython).

Keywords: algorithms, expression, tree, eval, inheritance, cdef class, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class Expr:
    """Base expression node."""

    cpdef double eval(self):
        return 0.0


cdef class Literal(Expr):
    """Literal value expression."""
    cdef double value

    def __cinit__(self, double value):
        self.value = value

    cpdef double eval(self):
        return self.value


cdef class BinOp(Expr):
    """Binary operation expression (add or multiply)."""
    cdef Expr left
    cdef Expr right
    cdef int op

    def __cinit__(self, Expr left, Expr right, int op):
        self.left = left
        self.right = right
        self.op = op  # 0 = add, 1 = multiply

    cpdef double eval(self):
        cdef double lv = self.left.eval()
        cdef double rv = self.right.eval()
        if self.op == 0:
            return lv + rv
        else:
            return lv * rv


cdef tuple _build_tree(int depth, long long seed):
    """Deterministically build an expression tree."""
    cdef double val
    cdef int op
    cdef Expr left, right
    if depth <= 0:
        val = ((seed * <long long>2654435761 + 17) % 10000) / 100.0 - 50.0
        return Literal(val), seed + 1
    op = ((seed * <long long>1103515245 + 12345) >> 4) & 1
    left, seed = _build_tree(depth - 1, seed + 1)
    right, seed = _build_tree(depth - 1, seed + 1)
    return BinOp(left, right, op), seed


@cython_benchmark(syntax="cy", args=(50000,))
def expression_eval(int n):
    """Build and evaluate n expression trees, sum results."""
    cdef double total = 0.0
    cdef int i
    cdef Expr tree

    for i in range(n):
        tree, _seed_out = _build_tree(4, <long long>i * 37)
        total += tree.eval()

    return total
