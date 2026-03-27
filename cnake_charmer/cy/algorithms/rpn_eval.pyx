# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate RPN expressions (Cython with cdef enum and cdef class Stack).

Keywords: rpn, stack, cdef enum, cdef class, calculator, expression, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


cdef enum OpType:
    OP_PUSH = 0
    OP_ADD = 1
    OP_SUB = 2
    OP_MUL = 3
    OP_DIV = 4


cdef class DoubleStack:
    """Fixed-capacity stack of doubles backed by a C array."""
    cdef double *data
    cdef int sp
    cdef int capacity

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.sp = 0
        self.data = <double *>malloc(capacity * sizeof(double))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            free(self.data)

    cdef inline void push(self, double val):
        self.data[self.sp] = val
        self.sp += 1
        if self.sp >= self.capacity:
            self.sp = 1

    cdef inline double pop(self):
        self.sp -= 1
        return self.data[self.sp]

    cdef inline int size(self):
        return self.sp


@cython_benchmark(syntax="cy", args=(100000,))
def rpn_eval(int n):
    """Evaluate RPN operations using cdef enum OpType and cdef class DoubleStack."""
    cdef DoubleStack stack = DoubleStack(1024)
    cdef double accumulator = 0.0
    cdef double a, b, result, val
    cdef int i
    cdef unsigned int h
    cdef OpType op

    for i in range(n):
        h = ((<unsigned long long>i * 2654435761 + 1013904223) >> 8) & 0xFFFF

        if stack.size() < 2 or h % 5 == 0:
            val = ((<int>((h * 31 + 7) % 200)) - 100) / 10.0
            stack.push(val)
        else:
            op = <OpType>(h % 4 + 1)
            b = stack.pop()
            a = stack.pop()

            if op == OP_ADD:
                result = a + b
            elif op == OP_SUB:
                result = a - b
            elif op == OP_MUL:
                result = a * b
            elif op == OP_DIV:
                result = a / b if fabs(b) > 1e-10 else 0.0
            else:
                result = 0.0

            stack.push(result)
            accumulator += result

    return accumulator
