# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Huffman-like encoding length computation (Cython-optimized).

Keywords: compression, huffman, frequency, encoding, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


cdef void _sort_nodes(long long *freqs, int *depths, int count):
    """Simple insertion sort for the node arrays by frequency."""
    cdef int i, j
    cdef long long tf
    cdef int td
    for i in range(1, count):
        tf = freqs[i]
        td = depths[i]
        j = i
        while j > 0 and freqs[j - 1] > tf:
            freqs[j] = freqs[j - 1]
            depths[j] = depths[j - 1]
            j -= 1
        freqs[j] = tf
        depths[j] = td


@cython_benchmark(syntax="cy", args=(1000000,))
def huffman_frequency(int n):
    """Compute sum of (frequency * depth) for Huffman-like encoding."""
    cdef int freq[26]
    cdef int i
    memset(freq, 0, 26 * sizeof(int))

    # Count character frequencies
    for i in range(n):
        freq[(i * 7 + 3) % 26] += 1

    # Collect non-zero frequencies
    # We'll use a priority-queue simulation with arrays
    # Each leaf starts at depth 0; when we merge two nodes, all leaves
    # under them get depth+1. We track this via a flat representation.
    #
    # Simpler approach: track (freq, leaf_list) as Python did.
    # But for full C, we use the "sum of merged frequencies" property:
    # total Huffman cost = sum of all internal node frequencies.
    # This is equivalent to sum(freq*depth) for leaves.

    # Collect non-zero freqs into a sortable array
    cdef long long *node_freqs = <long long *>malloc(26 * sizeof(long long))
    if not node_freqs:
        raise MemoryError()

    cdef int count = 0
    for i in range(26):
        if freq[i] > 0:
            node_freqs[count] = freq[i]
            count += 1

    cdef long long single_result = 0
    if count <= 1:
        if count == 1:
            single_result = node_freqs[0]
        free(node_freqs)
        return int(single_result)

    # Sort ascending
    cdef int ci, cj
    cdef long long tmp
    for ci in range(1, count):
        tmp = node_freqs[ci]
        cj = ci
        while cj > 0 and node_freqs[cj - 1] > tmp:
            node_freqs[cj] = node_freqs[cj - 1]
            cj -= 1
        node_freqs[cj] = tmp

    # Huffman cost = sum of all internal node weights
    # Each merge creates an internal node with weight = sum of two children
    cdef long long total_cost = 0
    cdef long long merged

    # Use two-queue approach for O(n) Huffman after sorting
    cdef long long *queue2 = <long long *>malloc(26 * sizeof(long long))
    if not queue2:
        free(node_freqs)
        raise MemoryError()
    cdef int q1_start = 0, q2_start = 0, q2_end = 0

    cdef long long a, b

    for i in range(count - 1):
        # Pick smallest from front of queue1 or queue2
        if q2_start < q2_end and (q1_start >= count or queue2[q2_start] <= node_freqs[q1_start]):
            a = queue2[q2_start]
            q2_start += 1
        else:
            a = node_freqs[q1_start]
            q1_start += 1

        if q2_start < q2_end and (q1_start >= count or queue2[q2_start] <= node_freqs[q1_start]):
            b = queue2[q2_start]
            q2_start += 1
        else:
            b = node_freqs[q1_start]
            q1_start += 1

        merged = a + b
        total_cost += merged
        queue2[q2_end] = merged
        q2_end += 1

    free(node_freqs)
    free(queue2)
    return int(total_cost)
