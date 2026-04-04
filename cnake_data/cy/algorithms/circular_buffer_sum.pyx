# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Accumulate windowed sums using a cdef class circular buffer.

Keywords: circular buffer, cdef class, __getitem__, __setitem__, __len__, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef class CircularBuffer:
    """Fixed-capacity circular buffer with __getitem__, __setitem__, __len__."""
    cdef double *data
    cdef int capacity
    cdef int head
    cdef int _size

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.head = 0
        self._size = 0
        self.data = <double *>malloc(capacity * sizeof(double))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            free(self.data)

    def __len__(self):
        return self._size

    def __getitem__(self, Py_ssize_t idx):
        """Get element by logical index (0 = oldest)."""
        if idx < 0 or idx >= self._size:
            raise IndexError("index out of range")
        cdef int start = (self.head - self._size + self.capacity) % self.capacity
        return self.data[(start + idx) % self.capacity]

    def __setitem__(self, Py_ssize_t idx, double val):
        """Set element by logical index."""
        if idx < 0 or idx >= self._size:
            raise IndexError("index out of range")
        cdef int start = (self.head - self._size + self.capacity) % self.capacity
        self.data[(start + idx) % self.capacity] = val

    cdef void push(self, double val):
        """Push a new value, overwriting oldest if full."""
        self.data[self.head] = val
        self.head = (self.head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    cdef double get_from_end(self, int offset):
        """Get element at offset from the most recent (0 = newest)."""
        cdef int idx = (self.head - 1 - offset + self.capacity) % self.capacity
        return self.data[idx]


@cython_benchmark(syntax="cy", args=(100000,))
def circular_buffer_sum(int n):
    """Push values into CircularBuffer and accumulate windowed sums."""
    cdef CircularBuffer buf = CircularBuffer(500)
    cdef double grand_total = 0.0
    cdef double val, window_sum
    cdef int i, j, window, k = 100

    for i in range(n):
        val = ((<long long>i * <long long>2654435761 + 7) % 10000) / 100.0
        buf.push(val)

        window = min(buf._size, k)
        window_sum = 0.0
        for j in range(window):
            window_sum += buf.get_from_end(j)

        grand_total += window_sum

    return grand_total
