"""Wildcard pattern matching against generated strings.

Keywords: dynamic programming, wildcard, pattern matching, string, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def wildcard_matching(n: int) -> int:
    """Count how many generated strings match pattern "D*?D".

    '*' matches zero or more characters, '?' matches exactly one.
    Strings: 5 chars each, s[j] = chr(65 + (i*j + 3) % 26) for j in 0..4.

    Args:
        n: Number of strings to test.

    Returns:
        Count of matching strings.
    """
    pattern = "D*?D"
    pat_len = len(pattern)
    count = 0

    for i in range(n):
        s = ""
        for j in range(5):
            s += chr(65 + (i * j + 3) % 26)

        # DP matching
        s_len = len(s)
        # dp[pi][si] = can pattern[pi:] match s[si:]
        # Use two rows
        prev = [False] * (s_len + 1)
        prev[s_len] = True  # empty pattern matches empty string

        for pi in range(pat_len - 1, -1, -1):
            curr = [False] * (s_len + 1)
            pc = pattern[pi]
            if pc == "*":
                # '*' can match empty (curr[si] = prev[si])
                # or one more char (curr[si] = curr[si+1])
                curr[s_len] = prev[s_len]
                for si in range(s_len - 1, -1, -1):
                    curr[si] = prev[si] or curr[si + 1]
            else:
                for si in range(s_len - 1, -1, -1):
                    if pc == "?" or pc == s[si]:
                        curr[si] = prev[si + 1]
            prev = curr

        if prev[0]:
            count += 1

    return count
