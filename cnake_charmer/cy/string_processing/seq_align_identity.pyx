# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sequence alignment identity computation (Cython-optimized).

Compares aligned DNA-like sequence pairs using C-level byte operations.

Keywords: sequence alignment, identity, DNA, bioinformatics, string processing, cython
"""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000, 15))
def seq_align_identity(int n, int gap_rate):
    """Compute alignment identity statistics over many sequence pairs."""
    cdef int p, i, seed, matches, gaps, aligned_len
    cdef int total_matches = 0
    cdef double total_identity_sum = 0.0
    cdef int total_gaps = 0
    cdef int n_pairs = 50
    cdef double identity_pct
    cdef char dash = b'-'

    cdef char *bases = b"ACGT"
    cdef char *seq1 = <char *>malloc(n * sizeof(char))
    cdef char *seq2 = <char *>malloc(n * sizeof(char))
    if not seq1 or not seq2:
        free(seq1)
        free(seq2)
        raise MemoryError()

    for p in range(n_pairs):
        # Build two aligned sequences deterministically
        for i in range(n):
            seed = p * 997 + i * 31
            if i % gap_rate == (p % gap_rate):
                seq1[i] = dash
                seq2[i] = bases[(seed * 7 + 3) % 4]
            elif i % gap_rate == ((p + 1) % gap_rate):
                seq1[i] = bases[seed % 4]
                seq2[i] = dash
            else:
                seq1[i] = bases[seed % 4]
                seq2[i] = bases[(seed * 7 + 3) % 4]

        # Compute identity stats
        matches = 0
        gaps = 0
        aligned_len = 0
        for i in range(n):
            if seq1[i] == dash or seq2[i] == dash:
                gaps += 1
            else:
                aligned_len += 1
                if seq1[i] == seq2[i]:
                    matches += 1

        identity_pct = <double>matches / <double>aligned_len if aligned_len > 0 else 0.0
        total_matches += matches
        total_identity_sum += identity_pct
        total_gaps += gaps

    free(seq1)
    free(seq2)

    return (total_matches, total_identity_sum, total_gaps)
