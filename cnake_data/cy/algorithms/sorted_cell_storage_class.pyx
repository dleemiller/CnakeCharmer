# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based sorted storage with nearest-value queries (Cython)."""

from libc.stdlib cimport malloc, free, realloc

from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class SortedCellStorage:
    cdef int *values
    cdef int size
    cdef int cap

    def __cinit__(self):
        self.size = 0
        self.cap = 1024
        self.values = <int *>malloc(self.cap * sizeof(int))
        if self.values == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.values != NULL:
            free(self.values)

    cdef void _grow(self) except *:
        cdef int new_cap = self.cap * 2
        cdef int *new_vals = <int *>realloc(self.values, new_cap * sizeof(int))
        if new_vals == NULL:
            raise MemoryError()
        self.values = new_vals
        self.cap = new_cap

    cdef void insert(self, int value) except *:
        cdef int lo = 0
        cdef int hi = self.size
        cdef int mid
        cdef int i

        while lo < hi:
            mid = (lo + hi) // 2
            if self.values[mid] < value:
                lo = mid + 1
            else:
                hi = mid

        if self.size == self.cap:
            self._grow()

        for i in range(self.size, lo, -1):
            self.values[i] = self.values[i - 1]
        self.values[lo] = value
        self.size += 1

    cdef int nearest(self, int target) noexcept nogil:
        cdef int n = self.size
        cdef int lo = 0
        cdef int hi = n
        cdef int mid
        cdef int a, b, da, db

        if n == 0:
            return 0

        while lo < hi:
            mid = (lo + hi) // 2
            if self.values[mid] < target:
                lo = mid + 1
            else:
                hi = mid

        if lo == 0:
            return self.values[0]
        if lo == n:
            return self.values[n - 1]

        a = self.values[lo - 1]
        b = self.values[lo]
        da = target - a
        db = b - target
        return a if da <= db else b


@cython_benchmark(syntax="cy", args=(5000, 220000, 29, 100003))
def sorted_cell_storage_class(int n_values, int n_queries, int seed, int mod):
    cdef SortedCellStorage st = SortedCellStorage()
    cdef int i, q, v, t, n
    cdef unsigned int s = 0
    cdef unsigned int near_sum = 0
    cdef unsigned long long uu

    for i in range(n_values):
        uu = (
            <unsigned long long>seed * 2654435761
            + <unsigned long long>i * 40507
            + <unsigned long long>(i % 11) * 97
        )
        v = <int>(uu % <unsigned long long>mod)
        st.insert(v)

    with nogil:
        for q in range(n_queries):
            uu = <unsigned long long>seed * 1664525 + <unsigned long long>q * 1013904223 + s
            t = <int>(uu % <unsigned long long>mod)
            n = st.nearest(t)
            near_sum += <unsigned int>n
            s = (s + <unsigned int>(n ^ t)) & MASK32

    return (st.size, near_sum & MASK32, s)
