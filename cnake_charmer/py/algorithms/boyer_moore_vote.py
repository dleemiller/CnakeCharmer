"""Heavy hitter detection using Boyer-Moore majority vote variant.

Keywords: algorithms, boyer-moore, majority vote, heavy hitter, frequency, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def boyer_moore_vote(n: int) -> tuple:
    """Detect top-2 heavy hitters using Boyer-Moore majority vote variant.

    Generates n deterministic elements and finds the two most frequent
    candidates using a two-pass approach.

    Args:
        n: Number of elements to process.

    Returns:
        Tuple of (candidate1, frequency1, candidate2).
    """
    # Phase 1: Find two candidates using extended Boyer-Moore vote
    cand1 = 0
    cand2 = 0
    count1 = 0
    count2 = 0

    for i in range(n):
        # Deterministic hash that stays in range, no overflow issues
        val = ((i * 17 + 3) ^ (i * 31 + 7)) % 50

        if val == cand1:
            count1 += 1
        elif val == cand2:
            count2 += 1
        elif count1 == 0:
            cand1 = val
            count1 = 1
        elif count2 == 0:
            cand2 = val
            count2 = 1
        else:
            count1 -= 1
            count2 -= 1

    # Phase 2: Count actual frequencies of candidates
    freq1 = 0
    freq2 = 0
    for i in range(n):
        val = ((i * 17 + 3) ^ (i * 31 + 7)) % 50
        if val == cand1:
            freq1 += 1
        elif val == cand2:
            freq2 += 1

    # Ensure cand1 has higher frequency
    if freq2 > freq1:
        cand1, cand2 = cand2, cand1
        freq1, freq2 = freq2, freq1

    return (cand1, freq1, cand2)
