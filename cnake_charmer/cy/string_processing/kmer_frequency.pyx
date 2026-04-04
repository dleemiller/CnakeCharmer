# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
K-mer (dinucleotide) frequency counting on a synthetic DNA sequence (Cython-optimized).

Keywords: string processing, kmer, dna, frequency, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def kmer_frequency(int n):
    """Count 2-mer frequencies in a synthetic DNA string using C arrays."""
    cdef int i
    cdef int most_freq_idx, most_freq_count
    cdef int counts[16]
    cdef unsigned char *seq = <unsigned char *>malloc(n * sizeof(unsigned char))

    if not seq:
        raise MemoryError("Failed to allocate sequence array")

    memset(counts, 0, 16 * sizeof(int))

    with nogil:
        # Fill sequence using LCG low-2-bits
        for i in range(n):
            seq[i] = (i * 1664525 + 1013904223) & 3

        # Count 2-mers
        for i in range(1, n):
            counts[seq[i - 1] * 4 + seq[i]] += 1

    most_freq_idx = 0
    most_freq_count = counts[0]
    for i in range(1, 16):
        if counts[i] > most_freq_count:
            most_freq_count = counts[i]
            most_freq_idx = i

    free(seq)
    return (n - 1, most_freq_idx, most_freq_count)
