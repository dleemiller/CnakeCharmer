# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Diagonal scan for identity segments in sequence alignment (Cython-optimized)."""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def diagonal_scan(int n):
    """Find all diagonal segments with identity score >= threshold using C arrays.

    Args:
        n: Length of both sequences.

    Returns:
        (total_segment_count, total_segment_length_sum) across all diagonals.
    """
    cdef int i, d, i_lo, i_hi, length, pos, score
    cdef int in_segment, seg_start, seg_len
    cdef long total_count, total_length
    cdef int window = 10
    cdef int threshold = 2
    cdef int *seq1
    cdef int *seq2
    cdef int *matches

    seq1 = <int *>malloc(n * sizeof(int))
    seq2 = <int *>malloc(n * sizeof(int))
    matches = <int *>malloc(n * sizeof(int))

    if seq1 == NULL or seq2 == NULL or matches == NULL:
        if seq1 != NULL: free(seq1)
        if seq2 != NULL: free(seq2)
        if matches != NULL: free(matches)
        raise MemoryError("Failed to allocate sequence arrays")

    with nogil:
        for i in range(n):
            seq1[i] = 65 + (i * 7 + 13) % 20
            seq2[i] = 65 + (i * 11 + 7) % 20

    total_count = 0
    total_length = 0

    with nogil:
        for d in range(-n // 4, n // 4 + 1):
            i_lo = -d if d < 0 else 0
            i_hi = n - d if d > 0 else n
            if i_hi > n:
                i_hi = n
            length = i_hi - i_lo
            if length < window:
                continue

            # Build match array for this diagonal
            for i in range(length):
                matches[i] = 1 if seq1[i_lo + i] == seq2[i_lo + i + d] else 0

            # Initialize window score
            score = 0
            for i in range(window):
                score += matches[i]

            in_segment = 0
            seg_start = 0

            for pos in range(length - window + 1):
                if pos > 0:
                    score -= matches[pos - 1]
                    score += matches[pos + window - 1]

                if score >= threshold:
                    if not in_segment:
                        in_segment = 1
                        seg_start = pos
                else:
                    if in_segment:
                        in_segment = 0
                        seg_len = (pos - 1 + window) - seg_start
                        total_count += 1
                        total_length += seg_len

            if in_segment:
                seg_len = (length - 1) - seg_start + 1
                total_count += 1
                total_length += seg_len

    free(seq1)
    free(seq2)
    free(matches)

    return (int(total_count), int(total_length))
