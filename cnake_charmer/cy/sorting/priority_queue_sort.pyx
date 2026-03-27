# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sort priority items using cdef class with __richcmp__ (Cython).

Keywords: sorting, priority queue, comparison, richcmp, cdef class, cython, benchmark
"""

from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cnake_charmer.benchmarks import cython_benchmark


cdef class PriorityItem:
    """Item with priority and value, comparable via __richcmp__."""
    cdef long long priority
    cdef long long value

    def __cinit__(self, long long priority, long long value):
        self.priority = priority
        self.value = value

    def __richcmp__(self, other, int op):
        cdef PriorityItem o = <PriorityItem>other
        if op == Py_LT:
            if self.priority != o.priority:
                return self.priority < o.priority
            return self.value < o.value
        elif op == Py_LE:
            if self.priority != o.priority:
                return self.priority < o.priority
            return self.value <= o.value
        elif op == Py_EQ:
            return self.priority == o.priority and self.value == o.value
        elif op == Py_NE:
            return self.priority != o.priority or self.value != o.value
        elif op == Py_GT:
            if self.priority != o.priority:
                return self.priority > o.priority
            return self.value > o.value
        elif op == Py_GE:
            if self.priority != o.priority:
                return self.priority > o.priority
            return self.value >= o.value
        return NotImplemented


@cython_benchmark(syntax="cy", args=(50000,))
def priority_queue_sort(int n):
    """Build n PriorityItem objects, sort them, return checksum of sorted order."""
    cdef list items = []
    cdef long long priority, value
    cdef long long checksum = 0
    cdef int i
    cdef PriorityItem item

    for i in range(n):
        priority = ((<long long>i * <long long>2654435761) ^ (<long long>i * <long long>1103515245 + 12345)) % 1000
        value = ((<long long>i * <long long>1664525 + <long long>1013904223) ^ (<long long>i * <long long>214013)) % 100000
        items.append(PriorityItem(priority, value))

    items.sort()

    for i in range(n):
        item = <PriorityItem>items[i]
        checksum = (checksum * 31 + item.priority * 1000000 + item.value) & <long long>0x7FFFFFFFFFFFFFFF
    return checksum
