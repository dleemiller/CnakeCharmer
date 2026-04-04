"""Count ways to segment a string into dictionary words.

Keywords: leetcode, word break, dynamic programming, segmentation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def word_break_count(n: int) -> tuple:
    """Count ways to break a deterministic string into dictionary words.

    Generates a string s of length n using s[i] = chr(97 + (i*7+3) % 5)
    for a 5-char alphabet (a-e). Dictionary contains all substrings of
    length 1-6 that appear in the first 100 characters. Uses DP to count
    total segmentations.

    Args:
        n: Length of the string.

    Returns:
        Tuple of (count_mod, max_word_len_used, num_dp_nonzero).
    """
    mod = 1000000007

    # Generate string
    s = [0] * n
    for i in range(n):
        s[i] = (i * 7 + 3) % 5

    # Build dictionary from substrings of first min(100, n) chars
    prefix_len = 100 if n >= 100 else n
    max_word = 6
    dictionary = set()
    for i in range(prefix_len):
        for length in range(1, max_word + 1):
            if i + length <= prefix_len:
                # Use tuple as hashable key
                word = tuple(s[i : i + length])
                dictionary.add(word)

    # DP: dp[i] = number of ways to segment s[0:i]
    dp = [0] * (n + 1)
    dp[0] = 1

    max_used = 0
    for i in range(1, n + 1):
        for length in range(1, max_word + 1):
            if length > i:
                break
            word = tuple(s[i - length : i])
            if word in dictionary:
                dp[i] = (dp[i] + dp[i - length]) % mod
                if dp[i - length] > 0 and length > max_used:
                    max_used = length

    # Count non-zero dp entries
    num_nonzero = 0
    for i in range(n + 1):
        if dp[i] > 0:
            num_nonzero += 1

    return (dp[n], max_used, num_nonzero)
