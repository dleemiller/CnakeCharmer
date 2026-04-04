# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based barcode index with Hamming-distance binning (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


cdef class BarcodeIndex:
    cdef int n_codes
    cdef int code_len
    cdef unsigned char *codes

    def __cinit__(self, int n_codes, int code_len, int seed):
        cdef int i, j
        cdef unsigned int x
        self.n_codes = n_codes
        self.code_len = code_len
        self.codes = <unsigned char *>malloc(n_codes * code_len * sizeof(unsigned char))
        if self.codes == NULL:
            raise MemoryError()

        for i in range(n_codes):
            x = <unsigned int>((seed + i * 37) & 0x7FFFFFFF)
            for j in range(code_len):
                x = (1103515245 * x + 12345 + j * 17) & 0x7FFFFFFF
                self.codes[i * code_len + j] = <unsigned char>(x & 3)

    def __dealloc__(self):
        if self.codes != NULL:
            free(self.codes)

    cdef void mutate_code(self, int idx, int edits, int salt, unsigned char *out) noexcept nogil:
        cdef int j, k, pos
        for j in range(self.code_len):
            out[j] = self.codes[idx * self.code_len + j]
        for k in range(edits):
            pos = (salt * 131 + k * 17) % self.code_len
            out[pos] = <unsigned char>((out[pos] + 1 + ((salt + k) & 1)) & 3)

    cdef int hamming_probe(self, unsigned char *probe, int idx) noexcept nogil:
        cdef int j
        cdef int d = 0
        cdef int off = idx * self.code_len
        for j in range(self.code_len):
            if probe[j] != self.codes[off + j]:
                d += 1
        return d


@cython_benchmark(syntax="cy", args=(420, 28, 3, 7, 41))
def barcode_hamming_bins_class(
    int n_codes,
    int code_len,
    int edits,
    int threshold,
    int seed,
):
    cdef BarcodeIndex idx = BarcodeIndex(n_codes, code_len, seed)
    cdef unsigned char *probe = <unsigned char *>malloc(code_len * sizeof(unsigned char))
    cdef int i, j, d, best, best_j
    cdef int matches = 0
    cdef long long total_dist = 0
    cdef long long nearest_sum = 0

    if probe == NULL:
        raise MemoryError()

    with nogil:
        for i in range(n_codes):
            idx.mutate_code(i, edits, seed + i * 19, probe)
            best = code_len + 1
            best_j = -1
            for j in range(n_codes):
                d = idx.hamming_probe(probe, j)
                total_dist += d
                if d < best:
                    best = d
                    best_j = j
            if best <= threshold:
                matches += 1
            nearest_sum += best_j

    free(probe)
    return (matches, total_dist, nearest_sum)
