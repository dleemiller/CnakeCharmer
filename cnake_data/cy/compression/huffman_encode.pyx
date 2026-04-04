# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build Huffman tree and encode a deterministic string (Cython-optimized).

Keywords: compression, huffman, encoding, tree, bitlength, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def huffman_encode(int n):
    """Build Huffman tree and compute encoding stats with typed variables."""
    cdef int freq[26]
    cdef int i
    memset(freq, 0, 26 * sizeof(int))

    for i in range(n):
        freq[(i * 17 + 3) % 26] += 1

    # Collect non-zero frequencies
    cdef int count = 0
    cdef long long *node_freqs = <long long *>malloc(26 * sizeof(long long))
    cdef int *node_ids = <int *>malloc(26 * sizeof(int))
    if not node_freqs or not node_ids:
        if node_freqs: free(node_freqs)
        if node_ids: free(node_ids)
        raise MemoryError()

    for i in range(26):
        if freq[i] > 0:
            node_freqs[count] = freq[i]
            node_ids[count] = i
            count += 1

    cdef int num_unique = count

    if count <= 1:
        free(node_freqs)
        free(node_ids)
        return (n, 1, 1)

    # Sort by frequency (insertion sort)
    cdef int j
    cdef long long tf
    cdef int ti
    for i in range(1, count):
        tf = node_freqs[i]
        ti = node_ids[i]
        j = i
        while j > 0 and node_freqs[j - 1] > tf:
            node_freqs[j] = node_freqs[j - 1]
            node_ids[j] = node_ids[j - 1]
            j -= 1
        node_freqs[j] = tf
        node_ids[j] = ti

    # Simulate Huffman merge tracking leaf depths
    cdef int *leaf_group = <int *>malloc(count * sizeof(int))
    cdef long long *group_freq = <long long *>malloc(count * sizeof(long long))
    cdef int *leaf_depth = <int *>malloc(count * sizeof(int))
    if not leaf_group or not group_freq or not leaf_depth:
        free(node_freqs); free(node_ids)
        if leaf_group: free(leaf_group)
        if group_freq: free(group_freq)
        if leaf_depth: free(leaf_depth)
        raise MemoryError()

    for i in range(count):
        leaf_group[i] = i
        group_freq[i] = node_freqs[i]
        leaf_depth[i] = 0

    cdef int min1_idx, min2_idx
    cdef long long min1_freq, min2_freq
    cdef int num_groups = count

    for _ in range(count - 1):
        # Find smallest group
        min1_idx = -1
        min1_freq = 0x7FFFFFFFFFFFFFFF
        for i in range(num_groups):
            if group_freq[i] >= 0 and group_freq[i] < min1_freq:
                min1_freq = group_freq[i]
                min1_idx = i
        group_freq[min1_idx] = -1  # temporarily mark

        min2_idx = -1
        min2_freq = 0x7FFFFFFFFFFFFFFF
        for i in range(num_groups):
            if group_freq[i] >= 0 and group_freq[i] < min2_freq:
                min2_freq = group_freq[i]
                min2_idx = i

        # Merge groups
        for i in range(count):
            if leaf_group[i] == min1_idx or leaf_group[i] == min2_idx:
                leaf_depth[i] += 1
                leaf_group[i] = min1_idx

        group_freq[min1_idx] = min1_freq + min2_freq
        group_freq[min2_idx] = -2  # permanently removed

    # Compute total bits and shortest code
    cdef int code_len[26]
    memset(code_len, 0, 26 * sizeof(int))
    for i in range(count):
        code_len[node_ids[i]] = leaf_depth[i]

    cdef long long total_bits = 0
    cdef int shortest = n
    for i in range(26):
        if freq[i] > 0:
            total_bits += <long long>freq[i] * code_len[i]
            if code_len[i] < shortest:
                shortest = code_len[i]

    free(node_freqs)
    free(node_ids)
    free(leaf_group)
    free(group_freq)
    free(leaf_depth)

    return (int(total_bits), num_unique, shortest)
