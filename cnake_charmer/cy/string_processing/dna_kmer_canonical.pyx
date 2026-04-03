# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Canonical DNA k-mer counting via reverse complement comparison (Cython-optimized).

Keywords: string processing, dna, kmer, reverse complement, canonical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcmp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def dna_kmer_canonical(int n):
    """Extract k-mers, canonicalize using char* and memcmp, count uniques."""
    cdef int k = 8
    cdef int i, j
    cdef int num_kmers = n - k + 1
    cdef long long checksum = 0
    cdef unsigned char *seq = <unsigned char *>malloc(n * sizeof(unsigned char))
    cdef unsigned char rc_buf[8]
    cdef unsigned char c
    cdef int cmp_result

    if seq == NULL:
        raise MemoryError("Failed to allocate sequence")

    # Generate deterministic DNA sequence: bases = "ACGT"
    cdef unsigned char bases[4]
    bases[0] = ord('A')
    bases[1] = ord('C')
    bases[2] = ord('G')
    bases[3] = ord('T')

    for i in range(n):
        seq[i] = bases[(i * 7 + 3) % 4]

    # Process each k-mer
    # Use a Python set for uniqueness (can't avoid this for correctness)
    canonical_set = set()

    for i in range(num_kmers):
        # Build reverse complement into rc_buf
        for j in range(k):
            c = seq[i + k - 1 - j]
            if c == ord('A'):
                rc_buf[j] = ord('T')
            elif c == ord('T'):
                rc_buf[j] = ord('A')
            elif c == ord('C'):
                rc_buf[j] = ord('G')
            else:  # G
                rc_buf[j] = ord('C')

        # Compare kmer vs revcomp using memcmp
        cmp_result = memcmp(&seq[i], rc_buf, k)

        if cmp_result <= 0:
            # kmer is canonical
            canonical_set.add(seq[i:i + k])
            checksum += seq[i]
        else:
            # revcomp is canonical
            canonical_set.add(rc_buf[:k])
            checksum += rc_buf[0]

    cdef int num_unique = len(canonical_set)
    cdef long long result_checksum = checksum

    free(seq)
    return (num_unique, result_checksum)
