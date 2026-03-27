# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count elements found via binary search in a sorted cdef class array.

Keywords: sorted array, binary search, cdef class, __getitem__, __len__, __contains__, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef class SortedArray:
    """Sorted integer array supporting __getitem__, __len__, and __contains__."""
    cdef long long *data
    cdef Py_ssize_t size

    def __cinit__(self, Py_ssize_t capacity):
        self.size = 0
        self.data = <long long *>malloc(capacity * sizeof(long long))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            free(self.data)

    cdef void append(self, long long val):
        self.data[self.size] = val
        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.size:
            raise IndexError("index out of range")
        return self.data[idx]

    def __contains__(self, long long target):
        """Binary search for target in sorted data."""
        cdef Py_ssize_t lo = 0
        cdef Py_ssize_t hi = self.size - 1
        cdef Py_ssize_t mid
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.data[mid] == target:
                return True
            elif self.data[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return False


@cython_benchmark(syntax="cy", args=(100000,))
def sorted_array_search(int n):
    """Build a SortedArray and search using __contains__."""
    cdef SortedArray arr = SortedArray(n)
    cdef int i
    cdef long long target
    cdef int found = 0

    for i in range(n):
        arr.append(<long long>i * 3 + 1)

    for i in range(n // 2):
        target = ((<long long>i * <long long>2654435761 + 17) >> 4) % (<long long>n * 4)
        if target in arr:
            found += 1

    return found
