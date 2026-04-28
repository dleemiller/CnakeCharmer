def collatz_sequence_length(n0):
    """Compute Collatz trajectory length including start value."""
    n = n0
    length = 1
    while True:
        if n & 1:
            if n == 1:
                break
            n = 3 * n + 1
        else:
            n >>= 1
        length += 1
    return length


def collatz_tail(n0, tail_size=20):
    """Return the last tail_size values from the Collatz trajectory."""
    n = n0
    seq = [n]
    while True:
        if n & 1:
            if n == 1:
                break
            n = 3 * n + 1
        else:
            n >>= 1
        seq.append(n)
    return seq[-tail_size:]


def collatz_strided_best(n_max, stride_offset, stride):
    """Search numbers n = stride_offset+1, +stride for max sequence length."""
    longest_len = 0
    longest_n0 = 0

    for n0 in range(stride_offset + 1, n_max + 1, stride):
        curr_len = collatz_sequence_length(n0)
        if curr_len > longest_len:
            longest_len = curr_len
            longest_n0 = n0

    return [longest_len, longest_n0, collatz_tail(longest_n0)]


def collatz_scan(n_max):
    """Single-process equivalent of the multiprocess strided search."""
    return collatz_strided_best(n_max, 0, 1)
