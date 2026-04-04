# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Accumulate values using @cython.final cdef class with typed methods (Cython).

Keywords: numerical, accumulator, final, operations, cdef class, cython, benchmark
"""

cimport cython
from cnake_data.benchmarks import cython_benchmark


@cython.final
cdef class Accumulator:
    """Final accumulator with typed add, mul_add, scale, and reset operations."""
    cdef double value
    cdef int count

    def __cinit__(self):
        self.value = 0.0
        self.count = 0

    cdef inline void add(self, double x):
        """Add x to the accumulated value."""
        self.value += x
        self.count += 1

    cdef inline void mul_add(self, double x, double y):
        """Add x * y to the accumulated value."""
        self.value += x * y
        self.count += 1

    cdef inline void scale(self, double factor):
        """Scale the accumulated value."""
        self.value *= factor

    cdef inline double get_value(self):
        """Return current accumulated value."""
        return self.value

    cdef inline int get_count(self):
        """Return operation count."""
        return self.count

    cdef inline void reset(self):
        """Reset accumulator."""
        self.value = 0.0
        self.count = 0


@cython_benchmark(syntax="cy", args=(500000,))
def final_accumulator(int n):
    """Accumulate n values with mixed operations, return final value."""
    cdef Accumulator acc = Accumulator()
    cdef double total = 0.0
    cdef double val, val2
    cdef int i, op

    for i in range(n):
        op = ((<long long>i * <long long>2654435761) >> 4) & 3
        val = ((<long long>i * <long long>1664525 + <long long>1013904223) % 10000) / 10000.0

        if op == 0:
            acc.add(val)
        elif op == 1:
            val2 = ((<long long>i * <long long>1103515245 + 12345) % 10000) / 10000.0
            acc.mul_add(val, val2)
        elif op == 2:
            acc.scale(0.999)
        else:
            total += acc.get_value()
            acc.reset()

    total += acc.get_value()
    return total
