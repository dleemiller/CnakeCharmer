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
        return (int(single_result), 0, count)

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

    # Compute max_depth: we need to rebuild with depth tracking
    # Use the same two-queue approach but track depths
    # Re-sort node_freqs... but we already freed. Instead, compute max_depth
    # using the property that max_depth = number of merges where a leaf participates.
    # Simpler: rebuild from scratch with depth tracking.
    # Actually, we can compute depths separately. Let's use a separate pass.

    # We need to recompute. Let's redo with depth tracking.
    # Re-collect frequencies
    cdef int *depths = <int *>malloc(26 * sizeof(int))
    cdef long long *nf2 = <long long *>malloc(26 * sizeof(long long))
    if not depths or not nf2:
        free(node_freqs)
        free(queue2)
        if depths: free(depths)
        if nf2: free(nf2)
        raise MemoryError()

    cdef int count2 = 0
    for i in range(26):
        if freq[i] > 0:
            nf2[count2] = freq[i]
            depths[count2] = 0
            count2 += 1

    # Sort by frequency (insertion sort)
    _sort_nodes(nf2, depths, count2)

    # Two-queue Huffman with depth tracking
    # We need to track depths for each original leaf through merges
    # This is complex with two-queue. Use simpler approach: simulate merges.
    # Actually, use the formula: for each merge, the two nodes merged get +1 depth.
    # We'll track per-leaf depths using a list-like approach in C.

    # Simpler: just count the number of unique symbols (count) and compute max_depth
    # by simulating the Huffman tree shape.
    # For a proper solution, let's track leaf depths through the merge process.

    # Use arrays of leaf-depth pairs. Since we have at most 26 leaves,
    # and at most 25 merges, we can afford a simple O(n^2) approach.

    # leaf_depths[i] = current depth of leaf i
    cdef int max_depth = 0
    cdef int num_symbols = count2

    # Simulate merges using sorted queue
    # After each merge, increase depth of all leaves in the merged groups
    # Track which leaves belong to which group using group IDs

    cdef int *leaf_group = <int *>malloc(count2 * sizeof(int))
    cdef long long *group_freq = <long long *>malloc(count2 * sizeof(long long))
    cdef int *group_id_map = <int *>malloc((count2 + count2) * sizeof(int))
    if not leaf_group or not group_freq or not group_id_map:
        free(nf2); free(depths); free(node_freqs); free(queue2)
        if leaf_group: free(leaf_group)
        if group_freq: free(group_freq)
        if group_id_map: free(group_id_map)
        raise MemoryError()

    # Each leaf starts in its own group (group i)
    cdef int num_groups = count2
    for i in range(count2):
        leaf_group[i] = i
        group_freq[i] = nf2[i]

    # leaf_depth tracks depth per leaf
    cdef int *leaf_depth = <int *>malloc(count2 * sizeof(int))
    if not leaf_depth:
        free(nf2); free(depths); free(node_freqs); free(queue2)
        free(leaf_group); free(group_freq); free(group_id_map)
        raise MemoryError()
    for i in range(count2):
        leaf_depth[i] = 0

    cdef int g1, g2, min1_idx, min2_idx
    cdef long long min1_freq, min2_freq

    for _ in range(count2 - 1):
        # Find two groups with smallest freq
        min1_idx = -1
        min1_freq = 0x7FFFFFFFFFFFFFFF
        for i in range(num_groups):
            if group_freq[i] >= 0 and group_freq[i] < min1_freq:
                min1_freq = group_freq[i]
                min1_idx = i

        # Mark first as taken temporarily
        group_freq[min1_idx] = -1

        min2_idx = -1
        min2_freq = 0x7FFFFFFFFFFFFFFF
        for i in range(num_groups):
            if group_freq[i] >= 0 and group_freq[i] < min2_freq:
                min2_freq = group_freq[i]
                min2_idx = i

        # Merge: all leaves in group min1_idx and min2_idx get depth+1
        # Assign all to min1_idx group
        for i in range(count2):
            if leaf_group[i] == min1_idx or leaf_group[i] == min2_idx:
                leaf_depth[i] += 1
                leaf_group[i] = min1_idx

        group_freq[min1_idx] = min1_freq + min2_freq
        group_freq[min2_idx] = -2  # permanently removed

    for i in range(count2):
        if leaf_depth[i] > max_depth:
            max_depth = leaf_depth[i]

    free(nf2)
    free(depths)
    free(leaf_group)
    free(group_freq)
    free(group_id_map)
    free(leaf_depth)
    free(node_freqs)
    free(queue2)
    return (int(total_cost), max_depth, num_symbols)
