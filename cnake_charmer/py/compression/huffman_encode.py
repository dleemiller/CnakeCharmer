"""Build Huffman tree and encode a deterministic string.

Keywords: compression, huffman, encoding, tree, bitlength, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def huffman_encode(n: int) -> tuple:
    """Build a Huffman tree for a deterministic string and compute encoding stats.

    Generates a string of n characters using s[i] = chr(65 + (i*17+3) % 26),
    builds a Huffman tree, assigns codes, and encodes the full string.

    Args:
        n: Length of the string to encode.

    Returns:
        Tuple of (total_encoded_bits, num_unique_chars, shortest_code_length).
    """
    # Count character frequencies
    freq = [0] * 26
    for i in range(n):
        freq[(i * 17 + 3) % 26] += 1

    # Collect non-zero frequencies with their character indices
    freq_list = []
    for i in range(26):
        if freq[i] > 0:
            freq_list.append((freq[i], i))
    freq_list.sort()

    num_unique = len(freq_list)

    if num_unique == 1:
        return (n, 1, 1)

    # Build Huffman tree using list-based priority queue
    # Each node: (freq, [(char_index, depth), ...])
    queue = [(f, [(idx, 0)]) for f, idx in freq_list]

    while len(queue) > 1:
        queue.sort(key=lambda x: x[0])
        left = queue[0]
        right = queue[1]
        queue = queue[2:]

        merged_leaves = []
        for idx, d in left[1]:
            merged_leaves.append((idx, d + 1))
        for idx, d in right[1]:
            merged_leaves.append((idx, d + 1))

        queue.append((left[0] + right[0], merged_leaves))

    # Extract code lengths per character
    code_len = [0] * 26
    for idx, d in queue[0][1]:
        code_len[idx] = d

    # Total encoded bits = sum of freq[i] * code_len[i]
    total_bits = 0
    shortest = n  # will be replaced
    for i in range(26):
        if freq[i] > 0:
            total_bits += freq[i] * code_len[i]
            if code_len[i] < shortest:
                shortest = code_len[i]

    return (total_bits, num_unique, shortest)
