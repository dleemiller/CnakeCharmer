# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Generate DNA kmers and compute bounded Hamming scan statistics (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 21f9a3240326d207eff6dda2a957d3f9116efabf
- filename: util.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef int _early_hamming(const char* a, const char* b, int k, int max_d) noexcept nogil:
    cdef int i, d = 0
    for i in range(k):
        if a[i] != b[i]:
            d += 1
            if d > max_d:
                break
    return d


cdef void _generate_dna(char* dna, int dna_len, unsigned int* state) noexcept nogil:
    cdef int i
    cdef unsigned int s
    s = state[0]
    for i in range(dna_len):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        if (s & 3) == 0:
            dna[i] = b'A'
        elif (s & 3) == 1:
            dna[i] = b'C'
        elif (s & 3) == 2:
            dna[i] = b'G'
        else:
            dna[i] = b'T'
    state[0] = s


cdef void _scan_kmers(
    const char* dna,
    int n,
    int k,
    int max_d,
    int* close_out,
    int* min_dist_out,
    unsigned int* checksum_out,
) noexcept nogil:
    cdef int i, d
    cdef int close = 0
    cdef int min_dist = k
    cdef unsigned int checksum = 0
    cdef unsigned int mask = 0xFFFFFFFF
    for i in range(1, n):
        d = _early_hamming(dna, dna + i, k, max_d)
        if d <= max_d:
            close += 1
        if d < min_dist:
            min_dist = d
        checksum = (checksum + <unsigned int>(d * (i + 17))) & mask
    close_out[0] = close
    min_dist_out[0] = min_dist
    checksum_out[0] = checksum


@cython_benchmark(syntax="cy", args=(40000, 11, 2))
def stack_kmer_hamming_scan(int dna_len, int k, int max_d):
    cdef unsigned int state = 123456789
    cdef int n = dna_len - k + 1
    cdef char *dna
    cdef int close = 0
    cdef int min_dist = k
    cdef unsigned int checksum = 0

    if dna_len <= 0 or k <= 0 or n <= 0:
        return (0, 0, 0, 0)

    dna = <char *>malloc(dna_len * sizeof(char))
    if not dna:
        raise MemoryError()

    with nogil:
        _generate_dna(dna, dna_len, &state)
        _scan_kmers(dna, n, k, max_d, &close, &min_dist, &checksum)

    free(dna)
    return (n, close, min_dist, checksum)
