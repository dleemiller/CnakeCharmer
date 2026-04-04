"""Diagonal scan for identity segments in sequence alignment.

Keywords: string processing, sequence alignment, diagonal, identity, sliding window
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def diagonal_scan(n: int) -> tuple[int, int]:
    """Find all diagonal segments with identity score >= threshold.

    Generates two character arrays of length n:
      seq1[i] = 65 + (i * 7 + 13) % 20
      seq2[i] = 65 + (i * 11 + 7) % 20

    For each diagonal d in range(-n//4, n//4+1), with window=10 and threshold=2,
    slides a window along the diagonal counting positions where score >= threshold,
    then counts contiguous segments and their total length.

    Args:
        n: Length of both sequences.

    Returns:
        (total_segment_count, total_segment_length_sum) across all diagonals.
    """
    seq1 = [65 + (i * 7 + 13) % 20 for i in range(n)]
    seq2 = [65 + (i * 11 + 7) % 20 for i in range(n)]

    window = 10
    threshold = 2
    total_count = 0
    total_length = 0

    for d in range(-n // 4, n // 4 + 1):
        i_lo = max(0, -d)
        i_hi = min(n, n - d)
        length = i_hi - i_lo
        if length < window:
            continue

        # Build match array for this diagonal
        matches = [1 if seq1[i] == seq2[i + d] else 0 for i in range(i_lo, i_hi)]

        # Initialize window score for first window
        score = sum(matches[:window])
        in_segment = False
        seg_start = 0

        for pos in range(length - window + 1):
            if pos > 0:
                score -= matches[pos - 1]
                score += matches[pos + window - 1]

            if score >= threshold:
                if not in_segment:
                    in_segment = True
                    seg_start = pos
            else:
                if in_segment:
                    in_segment = False
                    seg_len = (pos - 1 + window) - seg_start
                    total_count += 1
                    total_length += seg_len

        if in_segment:
            seg_len = (length - 1) - seg_start + 1
            total_count += 1
            total_length += seg_len

    return (total_count, total_length)
