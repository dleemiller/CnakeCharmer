"""Longest substring without repeating characters.

Keywords: leetcode, sliding window, substring, no repeat, hash set, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def longest_substring_no_repeat(n: int) -> tuple:
    """Find the longest substring without repeating characters.

    Generates a deterministic string of length n using characters from a
    pool of 128 ASCII characters, then finds the length and starting index
    of the longest substring with all unique characters.

    Args:
        n: Length of the generated string.

    Returns:
        Tuple of (max length found, starting index of that substring,
        total number of window adjustments).
    """
    # Generate deterministic character codes in range [0, 128)
    # We work directly with integer codes for consistency
    codes = [0] * n
    for i in range(n):
        codes[i] = ((i * 2654435761) & 0xFFFFFFFF) % 128

    max_len = 0
    max_start = 0
    adjustments = 0

    # Sliding window with last-seen index tracking
    last_seen = [-1] * 128
    left = 0

    for right in range(n):
        c = codes[right]
        if last_seen[c] >= left:
            left = last_seen[c] + 1
            adjustments += 1
        last_seen[c] = right
        window_len = right - left + 1
        if window_len > max_len:
            max_len = window_len
            max_start = left

    return (max_len, max_start, adjustments)
