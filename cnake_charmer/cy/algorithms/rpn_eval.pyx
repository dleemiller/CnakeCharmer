# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate RPN expressions (Cython with cdef enum and stack-allocated C array).

Keywords: rpn, stack, cdef enum, calculator, expression, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


cdef enum OpType:
    OP_PUSH = 0
    OP_ADD = 1
    OP_SUB = 2
    OP_MUL = 3
    OP_DIV = 4


@cython_benchmark(syntax="cy", args=(100000,))
def rpn_eval(int n):
    """Evaluate RPN operations using cdef enum OpType and an inline C array stack."""
    cdef double stack[1024]
    cdef int sp = 0
    cdef double accumulator = 0.0
    cdef double a, b, result, val
    cdef int i
    cdef unsigned int h
    cdef OpType op

    with nogil:
        for i in range(n):
            h = ((<unsigned long long>i * <unsigned long long>2654435761 + <unsigned long long>1013904223) >> 8) & 0xFFFF

            if sp < 2 or h % 5 == 0:
                val = ((<int>((h * 31 + 7) % 200)) - 100) / 10.0
                stack[sp] = val
                sp += 1
                if sp >= 1024:
                    sp = 1
            else:
                op = <OpType>(h % 4 + 1)
                sp -= 1
                b = stack[sp]
                sp -= 1
                a = stack[sp]

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

                stack[sp] = result
                sp += 1
                accumulator += result

    return accumulator
