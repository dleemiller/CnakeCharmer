# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Profile fixed-width k-mer counts over generated DNA text (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 62e948730a990693030f1347641dec71b649050b
- filename: knucleotide.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(180000, 6, 31))
def stack2_kmer_profile(int dna_length, int motif_width, int seed_tag):
    cdef unsigned int state
    cdef unsigned char *dna_codes
    cdef int table_size, mask
    cdef int *counts
    cdef int idx, code, top_idx = 0, top_count, val, distinct = 0
    cdef unsigned int checksum = 0

    if motif_width <= 0 or dna_length < motif_width:
        return (0, 0, 0, 0)

    state = <unsigned int>((2463534242 + seed_tag * 1223) & 0xFFFFFFFF)
    dna_codes = <unsigned char *>malloc(dna_length * sizeof(unsigned char))
    table_size = 1 << (2 * motif_width)
    counts = <int *>malloc(table_size * sizeof(int))

    if not dna_codes or not counts:
        free(dna_codes)
        free(counts)
        raise MemoryError()

    for idx in range(dna_length):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        dna_codes[idx] = <unsigned char>(state & 3)

    for idx in range(table_size):
        counts[idx] = 0

    mask = table_size - 1
    code = 0
    for idx in range(motif_width):
        code = (code << 2) | dna_codes[idx]
    counts[code] += 1

    for idx in range(motif_width, dna_length):
        code = ((code << 2) & mask) | dna_codes[idx]
        counts[code] += 1

    top_count = counts[0]
    for idx in range(table_size):
        val = counts[idx]
        if val != 0:
            distinct += 1
            checksum = (checksum + <unsigned int>(val * (idx + 7))) & 0xFFFFFFFF
            if val > top_count:
                top_count = val
                top_idx = idx

    free(dna_codes)
    free(counts)
    return (distinct, top_idx, top_count, checksum)
