# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find kth smallest element using a cdef class max-heap (Cython).

Keywords: heap, kth smallest, cdef class, cdef inline, max-heap, sorting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef class MaxHeap:
    """Max-heap with fixed capacity, using cdef inline methods."""
    cdef long long *data
    cdef int size
    cdef int capacity

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.size = 0
        self.data = <long long *>malloc(capacity * sizeof(long long))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            free(self.data)

    cdef inline void push(self, long long val):
        """Push a value onto the heap."""
        self.data[self.size] = val
        self.size += 1
        self._sift_up(self.size - 1)

    cdef inline long long peek(self):
        """Return the maximum value without removing it."""
        return self.data[0]

    cdef inline void replace_top(self, long long val):
        """Replace the top (max) element and restore heap property."""
        self.data[0] = val
        self._sift_down(0)

    cdef void _sift_up(self, int pos) noexcept:
        cdef int parent
        cdef long long tmp
        cdef long long *d = self.data
        while pos > 0:
            parent = (pos - 1) >> 1
            if d[pos] > d[parent]:
                tmp = d[pos]
                d[pos] = d[parent]
                d[parent] = tmp
                pos = parent
            else:
                break

    cdef void _sift_down(self, int pos) noexcept:
        cdef int left, right, largest
        cdef long long tmp
        cdef long long *d = self.data
        cdef int sz = self.size
        while True:
            left = 2 * pos + 1
            right = 2 * pos + 2
            largest = pos
            if left < sz and d[left] > d[largest]:
                largest = left
            if right < sz and d[right] > d[largest]:
                largest = right
            if largest != pos:
                tmp = d[pos]
                d[pos] = d[largest]
                d[largest] = tmp
                pos = largest
            else:
                break


@cython_benchmark(syntax="cy", args=(100000,))
def heap_kth_smallest(int n):
    """Stream n values through a cdef class MaxHeap to find kth smallest."""
    cdef int k = 100
    cdef MaxHeap heap = MaxHeap(k)
    cdef long long result_sum = 0
    cdef long long val
    cdef int i

    for i in range(n):
        val = ((<long long>i * <long long>2654435761 + 17) ^ (<long long>i * <long long>1103515245)) % 1000000

        if heap.size < k:
            heap.push(val)
            if heap.size == k:
                result_sum += heap.peek()
        else:
            if val < heap.peek():
                heap.replace_top(val)
            result_sum += heap.peek()

    return result_sum
