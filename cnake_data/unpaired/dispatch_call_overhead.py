"""Simple def/cdef/cpdef-style dispatch loop benchmark facsimile."""

from __future__ import annotations


class A:
    def d(self):
        return 0

    def c(self):
        return 0

    def p(self):
        return 0

    def test_def(self, num):
        while num > 0:
            self.d()
            num -= 1

    def test_cdef(self, num):
        while num > 0:
            self.c()
            num -= 1

    def test_cpdef(self, num):
        while num > 0:
            self.p()
            num -= 1


def fib(_n):
    import numpy as np

    a = np.array([1, 1, 100], dtype=np.int64)
    return a.sum()
