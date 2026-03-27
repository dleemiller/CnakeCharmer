# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Clamped value container with @property and @value.setter.

Keywords: numerical, property, setter, clamped, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class BoundedValue:
    """Value clamped to [lo, hi] via property setter."""
    cdef double _lo
    cdef double _hi
    cdef double _value

    def __cinit__(self, double lo, double hi):
        self._lo = lo
        self._hi = hi
        self._value = lo

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, double v):
        if v < self._lo:
            self._value = self._lo
        elif v > self._hi:
            self._value = self._hi
        else:
            self._value = v


@cython_benchmark(syntax="cy", args=(100000,))
def property_bounded_value(int n):
    """Push n values through a bounded container."""
    cdef BoundedValue bv = BoundedValue(-100.0, 100.0)
    cdef double total = 0.0
    cdef double raw
    cdef long long seed
    cdef int i

    for i in range(n):
        seed = (
            <long long>i * <long long>2654435761 + 17
        ) & 0x7FFFFFFF
        raw = (seed % 1000) / 2.0 - 250.0
        bv.value = raw
        total += bv._value

    return total
