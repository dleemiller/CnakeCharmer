# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Ring buffer running mean computation (Cython with cdef class).

Keywords: ring buffer, circular buffer, running mean, cdef class, extension type, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef class RingBuffer:
    """Fixed-capacity circular buffer with O(1) push and running total."""
    cdef double *data
    cdef int capacity
    cdef int head
    cdef int size
    cdef double total

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.head = 0
        self.size = 0
        self.total = 0.0
        self.data = <double *>malloc(capacity * sizeof(double))
        if not self.data:
            raise MemoryError()
        cdef int i
        for i in range(capacity):
            self.data[i] = 0.0

    def __dealloc__(self):
        if self.data:
            free(self.data)

    cdef double push_and_mean(self, double val):
        """Push a value and return the current mean."""
        if self.size == self.capacity:
            self.total -= self.data[self.head]
        else:
            self.size += 1
        self.data[self.head] = val
        self.total += val
        self.head = (self.head + 1) % self.capacity
        return self.total / self.size


@cython_benchmark(syntax="cy", args=(100000,))
def ring_buffer_mean(int n):
    """Push n values into a cdef class ring buffer and accumulate running means."""
    cdef RingBuffer rb = RingBuffer(1000)
    cdef double mean_sum = 0.0
    cdef double val
    cdef int i

    for i in range(n):
        val = (i * 7 + 13) % 10007 / 100.0
        mean_sum += rb.push_and_mean(val)

    return mean_sum
