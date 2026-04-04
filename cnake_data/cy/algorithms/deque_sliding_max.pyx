# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sliding window maximum using a cdef class monotonic deque (Cython).

Keywords: sliding window, maximum, cdef class, monotonic deque, algorithms, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef class IntDeque:
    """Fixed-capacity deque of ints backed by a C array."""
    cdef int *data
    cdef int head
    cdef int tail
    cdef int capacity

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.data = <int *>malloc(capacity * sizeof(int))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            free(self.data)

    cdef inline bint empty(self):
        return self.head == self.tail

    cdef inline int front(self):
        return self.data[self.head]

    cdef inline int back(self):
        return self.data[self.tail - 1]

    cdef inline void push_back(self, int val):
        self.data[self.tail] = val
        self.tail += 1

    cdef inline void pop_front(self):
        self.head += 1

    cdef inline void pop_back(self):
        self.tail -= 1


@cython_benchmark(syntax="cy", args=(100000,))
def deque_sliding_max(int n):
    """Compute sliding window maximum using a cdef class IntDeque."""
    cdef int k = 1000
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i
    cdef long long tmp
    for i in range(n):
        tmp = ((<long long>i * <long long>2654435761 + 13) ^ (<long long>i >> 3))
        arr[i] = <int>(tmp % 1000000)

    cdef IntDeque dq = IntDeque(n)
    cdef long long max_sum = 0

    for i in range(n):
        # Remove elements outside window
        while not dq.empty() and dq.front() <= i - k:
            dq.pop_front()

        # Remove smaller elements from back
        while not dq.empty() and arr[dq.back()] <= arr[i]:
            dq.pop_back()

        dq.push_back(i)

        if i >= k - 1:
            max_sum += arr[dq.front()]

    free(arr)
    return max_sum
