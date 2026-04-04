"""Find the longest palindromic substring in a deterministic string.

Keywords: leetcode, palindrome, substring, expand around center, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(20000,))
def longest_palindrome_substring(n: int) -> tuple:
    """Find longest palindromic substring using expand-around-center.

    Generates s[i] = (i*i + 3*i + 7) % 8 for an 8-symbol alphabet
    (quadratic produces repeating patterns with palindromes),
    then finds the longest palindromic substring.

    Args:
        n: Length of the string.

    Returns:
        Tuple of (max_palindrome_length, start_position, char_code_at_center).
    """
    # Generate string as integer array
    s = [0] * n
    for i in range(n):
        s[i] = (i * i + 3 * i + 7) % 8

    max_len = 1
    best_start = 0

    for center in range(n):
        # Odd-length palindromes
        left = center
        right = center
        while left >= 0 and right < n and s[left] == s[right]:
            length = right - left + 1
            if length > max_len:
                max_len = length
                best_start = left
            left -= 1
            right += 1

        # Even-length palindromes
        left = center
        right = center + 1
        while left >= 0 and right < n and s[left] == s[right]:
            length = right - left + 1
            if length > max_len:
                max_len = length
                best_start = left
            left -= 1
            right += 1

    # Character at center of best palindrome
    center_idx = best_start + max_len // 2
    center_char = s[center_idx]

    return (max_len, best_start, center_char)
