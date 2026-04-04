"""Count total palindromic substrings and find longest using Manacher's algorithm.

Keywords: string processing, palindrome, manacher, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def manacher(n: int) -> tuple:
    """Find longest palindrome and count all palindromic substrings.

    String generated using xorshift PRNG (seed=42) over 3-symbol alphabet
    for meaningful palindrome occurrences.

    Args:
        n: Length of the string.

    Returns:
        Tuple of (max_length, center_pos, total_palindrome_count).
    """
    if n <= 0:
        return (0, 0, 0)

    # Build the string using xorshift PRNG
    s = [None] * n
    seed = 42
    for i in range(n):
        seed ^= (seed << 13) & 0xFFFFFFFF
        seed ^= (seed >> 17) & 0xFFFFFFFF
        seed ^= (seed << 5) & 0xFFFFFFFF
        s[i] = seed % 3

    # Transform: insert sentinel (99) between chars and at ends
    t_len = 2 * n + 1
    t = [99] * t_len
    for i in range(n):
        t[2 * i + 1] = s[i]

    # Manacher's algorithm
    p = [0] * t_len
    center = 0
    right = 0

    for i in range(t_len):
        mirror = 2 * center - i
        if i < right:
            p[i] = min(right - i, p[mirror])

        # Expand
        a = i + p[i] + 1
        b = i - p[i] - 1
        while a < t_len and b >= 0 and t[a] == t[b]:
            p[i] += 1
            a += 1
            b -= 1

        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Find max palindrome length and its center position in original string
    max_len = 0
    max_center_t = 0
    total = 0
    for i in range(t_len):
        total += (p[i] + 1) // 2
        if p[i] > max_len:
            max_len = p[i]
            max_center_t = i

    # Convert center from transformed to original index
    center_pos = max_center_t // 2

    return (max_len, center_pos, total)
