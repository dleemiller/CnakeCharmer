# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based mixed-type distance accumulator (HEOM-style, Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef class HeomAccumulator:
    cdef int n_cols
    cdef int cat_stride
    cdef unsigned char *is_cat

    def __cinit__(self, int n_cols, int cat_stride):
        cdef int i
        self.n_cols = n_cols
        self.cat_stride = cat_stride
        self.is_cat = <unsigned char *>malloc(n_cols * sizeof(unsigned char))
        if self.is_cat == NULL:
            raise MemoryError()
        for i in range(n_cols):
            self.is_cat[i] = 1 if (i % cat_stride) == 0 else 0

    def __dealloc__(self):
        if self.is_cat != NULL:
            free(self.is_cat)

    cdef int distance_scaled(self, int *rows, int ra, int rb, int *spans) noexcept nogil:
        cdef int c
        cdef int total = 0
        cdef int a, b, da
        for c in range(self.n_cols):
            a = rows[ra * self.n_cols + c]
            b = rows[rb * self.n_cols + c]
            if self.is_cat[c]:
                if a != b:
                    total += 1000
            else:
                da = a - b
                if da < 0:
                    da = -da
                total += (da * 1000) // spans[c]
        return total


@cython_benchmark(syntax="cy", args=(550, 18, 4, 23))
def heom_distance_accumulator_class(int n_rows, int n_cols, int cat_stride, int seed):
    cdef int *rows = <int *>malloc(n_rows * n_cols * sizeof(int))
    cdef int *mins = <int *>malloc(n_cols * sizeof(int))
    cdef int *maxs = <int *>malloc(n_cols * sizeof(int))
    cdef int *spans = <int *>malloc(n_cols * sizeof(int))
    cdef HeomAccumulator metric = HeomAccumulator(n_cols, cat_stride)
    cdef int r, c, v, base
    cdef long long total = 0
    cdef int cat_mismatches = 0
    cdef int last = 0

    if rows == NULL or mins == NULL or maxs == NULL or spans == NULL:
        free(rows)
        free(mins)
        free(maxs)
        free(spans)
        raise MemoryError()

    with nogil:
        for c in range(n_cols):
            mins[c] = 10**9
            maxs[c] = -10**9

        for r in range(n_rows):
            base = seed * 104729 + r * 8191
            for c in range(n_cols):
                v = (base + c * 131 + (r * c) * 17) % 251
                rows[r * n_cols + c] = v
                if v < mins[c]:
                    mins[c] = v
                if v > maxs[c]:
                    maxs[c] = v

        for c in range(n_cols):
            spans[c] = maxs[c] - mins[c]
            if spans[c] < 1:
                spans[c] = 1

        for r in range(1, n_rows):
            last = metric.distance_scaled(rows, r - 1, r, spans)
            total += last
            c = 0
            while c < n_cols:
                if rows[(r - 1) * n_cols + c] != rows[r * n_cols + c]:
                    cat_mismatches += 1
                c += cat_stride

    free(rows)
    free(mins)
    free(maxs)
    free(spans)
    return (total, cat_mismatches, last)
