"""Huffman-like encoding length computation.

Keywords: compression, huffman, frequency, encoding, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def huffman_frequency(n: int) -> int:
    """Compute sum of (frequency * depth) for Huffman-like encoding.

    Generates s[i] = chr(65 + (i * 7 + 3) % 26), computes character
    frequencies, then simulates Huffman tree construction by repeatedly
    merging the two smallest frequencies. Returns the sum of all
    (freq * depth) values, which represents the total encoded bit length.

    Args:
        n: Length of the string.

    Returns:
        Total encoded length (sum of freq * depth for all characters).
    """
    # Count character frequencies
    freq = [0] * 26
    for i in range(n):
        freq[(i * 7 + 3) % 26] += 1

    # Filter non-zero frequencies and pair with char index
    nodes = []
    for i in range(26):
        if freq[i] > 0:
            nodes.append((freq[i], i, 0))  # (frequency, id, depth)

    # Simulate Huffman tree building:
    # Repeatedly merge two smallest, tracking depth for each original char
    node_id = 26  # For internal nodes

    # Use a simple sorted list approach
    nodes.sort()

    while len(nodes) > 1:
        # Pop two smallest
        left = nodes[0]
        right = nodes[1]
        nodes = nodes[2:]

        # Increase depth for all original chars under these nodes
        # We track this by maintaining (freq, id, depth) and accumulating
        merged_freq = left[0] + right[0]

        # Insert merged node maintaining sorted order
        new_node = (merged_freq, node_id, 0)
        node_id += 1

        # Binary insert
        inserted = False
        for i in range(len(nodes)):
            if nodes[i][0] >= merged_freq:
                nodes.insert(i, new_node)
                inserted = True
                break
        if not inserted:
            nodes.append(new_node)

    # To get actual depths, rebuild using a different approach:
    # Build the tree structure and compute depths
    freq_list = []
    for i in range(26):
        if freq[i] > 0:
            freq_list.append((freq[i], i))
    freq_list.sort()

    if len(freq_list) == 1:
        return freq_list[0][0]

    # Use a list-based priority queue simulation
    # Each entry: (combined_freq, [list of (original_freq, depth)])
    queue = [(f, [(f, 0)]) for f, idx in freq_list]

    while len(queue) > 1:
        queue.sort(key=lambda x: x[0])
        left = queue[0]
        right = queue[1]
        queue = queue[2:]

        # Merge: increase depth of all leaves by 1
        merged_leaves = []
        for f, d in left[1]:
            merged_leaves.append((f, d + 1))
        for f, d in right[1]:
            merged_leaves.append((f, d + 1))

        merged_freq = left[0] + right[0]
        queue.append((merged_freq, merged_leaves))

    # Sum freq * depth
    total = 0
    for f, d in queue[0][1]:
        total += f * d

    return total
