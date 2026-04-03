def high_scoring_segments(n):
    """Find high-scoring contiguous segments in a sequence of scores.

    Generates a deterministic sequence of n integer scores (some positive,
    some negative), then scans for segments where the cumulative score
    exceeds a threshold, resetting when the score drops below zero or
    drops too far from the segment maximum. This is a variant of the
    maximum subarray problem used in genomic sequence analysis.

    Args:
        n: Length of the score sequence.

    Returns:
        (num_segments, total_segment_length, max_segment_score) as a tuple.
    """
    # Generate deterministic scores: mix of positive and negative
    scores = [0] * n
    for i in range(n):
        val = ((i * 37 + 13) % 100) - 55  # range roughly -55 to +44
        scores[i] = val

    thresh = 50
    max_dropoff = 30

    cur_score = 0
    max_score = 0
    start = 0
    end = 0

    segments = []

    for i in range(n):
        cur_score += scores[i]
        if cur_score >= max_score:
            max_score = cur_score
            end = i + 1

        if cur_score < 0 or cur_score < max_score - max_dropoff:
            if max_score >= thresh:
                segments.append((start, end, max_score))
            max_score = 0
            cur_score = 0
            start = i + 1
            end = i + 1

    # Handle final segment
    if max_score >= thresh:
        segments.append((start, end, max_score))

    num_segments = len(segments)
    total_length = sum(e - s for s, e, _ in segments)
    max_seg_score = max((sc for _, _, sc in segments), default=0)

    return (num_segments, total_length, max_seg_score)
