# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based queue relay workload with push/pop transitions (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class RelayQueue:
    cdef int *buf
    cdef int head
    cdef int tail
    cdef int size
    cdef int capacity

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.buf = <int *>malloc(capacity * sizeof(int))
        if self.buf == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.buf != NULL:
            free(self.buf)

    cdef void push(self, int value) noexcept nogil:
        if self.size == self.capacity:
            self.head += 1
            if self.head == self.capacity:
                self.head = 0
            self.size -= 1
        self.buf[self.tail] = value
        self.tail += 1
        if self.tail == self.capacity:
            self.tail = 0
        self.size += 1

    cdef int pop(self) noexcept nogil:
        cdef int out
        if self.size == 0:
            return -1
        out = self.buf[self.head]
        self.head += 1
        if self.head == self.capacity:
            self.head = 0
        self.size -= 1
        return out


@cython_benchmark(syntax="cy", args=(1024, 320000, 1337, 4095))
def queue_relay_state_class(int capacity, int rounds, int seed, int mask):
    cdef RelayQueue q = RelayQueue(capacity)
    cdef int t, x, y, last = 0
    cdef unsigned int checksum = 0
    cdef int hits = 0

    with nogil:
        for t in range(rounds):
            x = (seed * 1103515245 + t * 12345 + <int>checksum) & mask
            q.push(x)
            if (t & 3) != 0:
                y = q.pop()
                if y >= 0:
                    last = (y ^ t) & mask
                    checksum = (checksum + <unsigned int>last + <unsigned int>q.size) & MASK32
                    if (last & 63) == (t & 63):
                        hits += 1

    return (checksum, hits, q.size, last)
