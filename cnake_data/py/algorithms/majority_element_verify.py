"""Find and verify majority elements in multiple array segments using Boyer-Moore.

Keywords: algorithms, majority element, boyer moore, verification, voting, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def majority_element_verify(n: int) -> tuple:
    """Find majority elements across multiple segments and verify them.

    Splits a deterministic array of n elements into segments of size sqrt(n),
    finds a majority candidate in each segment using Boyer-Moore voting,
    verifies it, and accumulates results.

    Args:
        n: Total number of elements.

    Returns:
        Tuple of (verified_majority_count, total_majority_sum,
                  segments_with_no_majority).
    """
    # Compute integer sqrt
    seg_size = 1
    while (seg_size + 1) * (seg_size + 1) <= n:
        seg_size += 1

    verified_count = 0
    majority_sum = 0
    no_majority_count = 0

    seg_start = 0
    while seg_start < n:
        seg_end = seg_start + seg_size
        if seg_end > n:
            seg_end = n
        seg_len = seg_end - seg_start

        # Phase 1: Boyer-Moore voting to find candidate
        candidate = 0
        votes = 0
        for i in range(seg_start, seg_end):
            val = (i * 41 + 7) % 17
            if votes == 0:
                candidate = val
                votes = 1
            elif val == candidate:
                votes += 1
            else:
                votes -= 1

        # Phase 2: verification pass
        count = 0
        for i in range(seg_start, seg_end):
            val = (i * 41 + 7) % 17
            if val == candidate:
                count += 1

        if count * 2 > seg_len:
            verified_count += 1
            majority_sum += candidate
        else:
            no_majority_count += 1

        seg_start = seg_end

    return (verified_count, majority_sum, no_majority_count)
