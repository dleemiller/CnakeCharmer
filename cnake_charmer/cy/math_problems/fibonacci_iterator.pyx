# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum Fibonacci numbers using a cdef class iterator with __iter__/__next__.

Keywords: fibonacci, cdef class, __iter__, __next__, iterator, math, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class FibIterator:
    """Fibonacci sequence iterator yielding values mod 10^9+7."""
    cdef long long a
    cdef long long b
    cdef int remaining
    cdef long long mod

    def __cinit__(self, int count):
        self.a = 0
        self.b = 1
        self.remaining = count
        self.mod = 1000000007

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        cdef long long val = self.a
        cdef long long tmp = self.b
        self.b = (self.a + self.b) % self.mod
        self.a = tmp
        self.remaining -= 1
        return val


@cython_benchmark(syntax="cy", args=(100000,))
def fibonacci_iterator(int n):
    """Generate Fibonacci numbers via FibIterator and compute weighted sum."""
    cdef FibIterator fib = FibIterator(n)
    cdef long long total = 0
    cdef long long val
    cdef long long MOD = 1000000007
    cdef int i = 0

    for val in fib:
        total = (total + val * ((i % 256) + 1)) % MOD
        i += 1

    return total
