# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build and iterate a cdef class linked list with __iter__ and __next__.

Keywords: linked list, cdef class, __iter__, __next__, iterator, algorithms, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef class Node:
    """Singly linked list node."""
    cdef double value
    cdef Node next_node

    def __cinit__(self, double value):
        self.value = value
        self.next_node = None


cdef class LinkedList:
    """Singly linked list supporting iteration via __iter__/__next__."""
    cdef Node head
    cdef Node tail
    cdef Node _iter_current

    def __cinit__(self):
        self.head = None
        self.tail = None
        self._iter_current = None

    cdef void append(self, double value):
        cdef Node node = Node(value)
        if self.tail is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next_node = node
            self.tail = node

    def __iter__(self):
        self._iter_current = self.head
        return self

    def __next__(self):
        if self._iter_current is None:
            raise StopIteration
        cdef double val = self._iter_current.value
        self._iter_current = self._iter_current.next_node
        return val

    cdef double c_sum(self):
        """Traverse the list at C level, summing values without Python boxing."""
        cdef Node node = self.head
        cdef double total = 0.0
        while node is not None:
            total += node.value
            node = node.next_node
        return total

    cdef double c_sum_squares(self):
        """Traverse the list at C level, summing val*val without Python boxing."""
        cdef Node node = self.head
        cdef double sq_total = 0.0
        cdef double v
        while node is not None:
            v = node.value
            sq_total += v * v
            node = node.next_node
        return sq_total


@cython_benchmark(syntax="cy", args=(50000,))
def linked_list_sum(int n):
    """Build a LinkedList and sum via C-level traversal methods."""
    cdef LinkedList lst = LinkedList()
    cdef int i
    cdef double val

    for i in range(n):
        val = ((<long long>i * <long long>2654435761 + 13) % 10000) / 100.0
        lst.append(val)

    # C-level traversals: no Python iterator boxing
    return lst.c_sum() + lst.c_sum_squares()
