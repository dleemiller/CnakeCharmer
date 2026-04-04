# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""K-mer frequency counting for DNA-like sequences.

Keywords: kmer, frequency, dna, sequence analysis, substring counting, cython
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset

from cnake_data.benchmarks import cython_benchmark


cdef inline int encode_base(int b) nogil:
    """Encode A=0, C=1, G=2, T=3."""
    if b == 0:
        return 0
    elif b == 1:
        return 1
    elif b == 2:
        return 2
    else:
        return 3


@cython_benchmark(syntax="cy", args=(5000,))
def kmer_frequency(int n):
    """Count k-mer frequencies in a deterministic sequence of length n.

    Args:
        n: Sequence length.

    Returns:
        Tuple of (num_unique_3mers, most_frequent_count, total_pairs).
    """
    cdef int i
    cdef int kmer3, kmer5
    cdef int num_unique = 0
    cdef int most_frequent = 0
    cdef long total_pairs = 0

    # Generate sequence as integer array (0-3)
    cdef int *seq = <int *>malloc(n * sizeof(int))
    if not seq:
        raise MemoryError()

    for i in range(n):
        seq[i] = (i * 7 + 13) % 4

    # Count 3-mers (4^3 = 64 possible)
    cdef int num_3mers = 64
    cdef int *freq3 = <int *>malloc(num_3mers * sizeof(int))
    if not freq3:
        free(seq)
        raise MemoryError()
    memset(freq3, 0, num_3mers * sizeof(int))

    for i in range(n - 2):
        kmer3 = seq[i] * 16 + seq[i + 1] * 4 + seq[i + 2]
        freq3[kmer3] += 1

    for i in range(num_3mers):
        if freq3[i] > 0:
            num_unique += 1
        if freq3[i] > most_frequent:
            most_frequent = freq3[i]

    # Count 5-mers (4^5 = 1024 possible)
    cdef int num_5mers = 1024
    cdef int *freq5 = <int *>malloc(num_5mers * sizeof(int))
    if not freq5:
        free(seq)
        free(freq3)
        raise MemoryError()
    memset(freq5, 0, num_5mers * sizeof(int))

    for i in range(n - 4):
        kmer5 = (seq[i] * 256 + seq[i + 1] * 64 +
                 seq[i + 2] * 16 + seq[i + 3] * 4 + seq[i + 4])
        freq5[kmer5] += 1

    cdef int cnt
    for i in range(num_5mers):
        cnt = freq5[i]
        total_pairs += <long>cnt * (cnt - 1) / 2

    free(seq)
    free(freq3)
    free(freq5)
    return (num_unique, most_frequent, total_pairs)
