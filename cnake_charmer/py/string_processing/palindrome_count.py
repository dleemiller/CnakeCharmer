"""Count all palindromic substrings using expand-around-center.

Keywords: string processing, palindrome, substring, expand center, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def palindrome_count(n: int) -> tuple:
    """Count all palindromic substrings in a deterministic string.

    Text: chr(97 + (i*7+3) % 26) for i in range(n).
    Uses expand-around-center for both odd and even length palindromes.
    Returns (total_count, longest_length, checksum) where checksum is
    sum of lengths of all palindromic substrings.

    Args:
        n: Length of the text.

    Returns:
        Tuple of (total count, longest palindrome length, length checksum).
    """
    # Build text as list of ordinals for fast comparison
    text = [97 + (i * 7 + 3) % 26 for i in range(n)]

    total_count = 0
    longest = 0
    length_sum = 0

    for center in range(n):
        # Odd-length palindromes
        left = center
        right = center
        while left >= 0 and right < n and text[left] == text[right]:
            plen = right - left + 1
            total_count += 1
            length_sum += plen
            if plen > longest:
                longest = plen
            left -= 1
            right += 1

        # Even-length palindromes
        left = center
        right = center + 1
        while left >= 0 and right < n and text[left] == text[right]:
            plen = right - left + 1
            total_count += 1
            length_sum += plen
            if plen > longest:
                longest = plen
            left -= 1
            right += 1

    return (total_count, longest, length_sum)
