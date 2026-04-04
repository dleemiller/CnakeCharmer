# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based ring queue push/pop workload metrics (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class Queue:
    cdef unsigned int *buf
    cdef int cap
    cdef int head
    cdef int tail
    cdef int size

    def __cinit__(self, int capacity):
        self.buf = <unsigned int *>malloc(capacity * sizeof(unsigned int))
        if not self.buf:
            raise MemoryError()
        self.cap = capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def __dealloc__(self):
        if self.buf != NULL:
            free(self.buf)


cdef void _run_queue(Queue q, int rounds, int seed, unsigned int *checksum_out, int *popped_out) noexcept nogil:
    cdef int i
    cdef unsigned int x, v
    cdef unsigned int checksum = 0
    cdef int popped = 0

    for i in range(rounds):
        x = (seed * 1664525 + i * 1013904223) & MASK32
        if q.size == q.cap:
            q.head = (q.head + 1) % q.cap
            q.size -= 1
        q.buf[q.tail] = x
        q.tail = (q.tail + 1) % q.cap
        q.size += 1

        if i & 1:
            v = q.buf[q.head]
            q.head = (q.head + 1) % q.cap
            q.size -= 1
            checksum = (checksum + (v & 0xFFFF)) & MASK32
            popped += 1

    checksum_out[0] = checksum
    popped_out[0] = popped


@cython_benchmark(syntax="cy", args=(512, 900000, 17))
def ring_queue_class_ops(int capacity, int rounds, int seed):
    cdef Queue q = Queue(capacity)
    cdef unsigned int checksum = 0
    cdef int popped = 0

    with nogil:
        _run_queue(q, rounds, seed, &checksum, &popped)

    return (checksum, popped, q.size)
