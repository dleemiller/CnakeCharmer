# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Sequence alignment identity computation for pairs of strings (Cython-optimized).

Keywords: string processing, alignment, identity, bioinformatics, sequence, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(20000,))
def alignment_identity(int n):
    """Compute alignment identity for n pairs of 128-char sequences.

    Args:
        n: Number of sequence pairs to compare.

    Returns:
        Tuple of (sum of identity fractions, count of high-identity pairs).
    """
    cdef int seq_len = 128
    cdef double total_identity = 0.0
    cdef int high_count = 0
    cdef int i, j, matches
    cdef double identity
    cdef int c1, c2

    # Use integer codes instead of chars for speed: A=0, C=1, G=2, T=3
    for i in range(n):
        matches = 0
        for j in range(seq_len):
            c1 = ((i * 31 + j * 7 + 3) * 2903) % 4
            c2 = ((i * 37 + j * 11 + 5) * 3079) % 4
            if c1 == c2:
                matches += 1
        identity = <double>matches / <double>seq_len
        total_identity += identity
        if identity > 0.5:
            high_count += 1

    return (total_identity, high_count)
