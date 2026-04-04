"""Find longest valid parentheses substring using stack-based DP.

Keywords: leetcode, parentheses, longest, valid, stack, dynamic programming, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def longest_valid_parentheses(n: int) -> tuple:
    """Find longest valid parentheses substring in a deterministic string of length n.

    String is generated deterministically: s[i] = '(' if (i*2654435761) % 3 != 0 else ')'.
    Uses stack-based O(n) algorithm. Also counts total number of maximal valid substrings.

    Args:
        n: Length of the parentheses string.

    Returns:
        Tuple of (max_length, start_pos, total_valid_substrings).
    """
    # Generate deterministic string (0 = '(', 1 = ')')
    # Use bit 16 of LCG for better quality randomness
    s = [0] * n
    lcg = 123456789
    for i in range(n):
        lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
        s[i] = (lcg >> 16) & 1

    # Stack-based approach to compute dp array
    # dp[i] = length of longest valid substring ending at i
    dp = [0] * n

    # Use a stack of indices (as a list)
    stack = [0] * (n + 1)
    sp = 0  # stack pointer
    stack[0] = -1
    sp = 1

    for i in range(n):
        if s[i] == 0:  # '('
            stack[sp] = i
            sp += 1
        else:  # ')'
            sp -= 1
            if sp <= 0:
                stack[0] = i
                sp = 1
            else:
                dp[i] = i - stack[sp - 1]

    max_length = 0
    start_pos = 0
    for i in range(n):
        if dp[i] > max_length:
            max_length = dp[i]
            start_pos = i - dp[i] + 1

    # Count total maximal valid substrings
    total_valid = 0
    i = 0
    while i < n:
        if dp[i] > 0:
            total_valid += 1
            # Skip to end of this valid substring
            # A valid substring ending at i has length dp[i]
            # Check if previous position also ends a valid substring that extends this
            i += 1
        else:
            i += 1

    # More precise: count contiguous valid blocks
    total_valid = 0
    in_valid = False
    for i in range(n):
        if dp[i] > 0:
            if not in_valid:
                total_valid += 1
                in_valid = True
        else:
            in_valid = False

    return (max_length, start_pos, total_valid)
