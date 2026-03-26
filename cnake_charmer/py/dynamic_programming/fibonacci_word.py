"""
Count occurrences of pattern "AB" in the nth Fibonacci word using DP.

Keywords: dynamic programming, fibonacci, word, string, pattern, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def fibonacci_word(n: int) -> int:
    """Count occurrences of 'AB' in the nth Fibonacci word.

    Fibonacci words: F(1)="A", F(2)="AB", F(n)=F(n-1)+F(n-2).
    Uses DP to track counts and boundary characters without building strings.

    Args:
        n: Which Fibonacci word to analyze.

    Returns:
        Count of "AB" occurrences mod 10^9+7.
    """
    MOD = 1000000007

    if n <= 0:
        return 0
    if n == 1:
        return 0  # "A" has no "AB"
    if n == 2:
        return 1  # "AB" has one "AB"

    # Track: count of "AB", last char, first char, length
    # F(n) = F(n-1) + F(n-2)
    # When concatenating, new "AB" can appear at the boundary
    # if last char of F(n-1) is 'A' and first char of F(n-2) is 'B'

    # prev2 = F(1) = "A": count=0, first='A', last='A'
    # prev1 = F(2) = "AB": count=1, first='A', last='B'
    count_prev2 = 0
    first_prev2 = 0  # 'A' = 0
    last_prev2 = 0  # 'A' = 0

    count_prev1 = 1
    first_prev1 = 0  # 'A' = 0
    last_prev1 = 1  # 'B' = 1

    for _ in range(3, n + 1):
        # F(i) = F(i-1) + F(i-2)
        # boundary: last of F(i-1) is 'A' and first of F(i-2) is 'B'
        boundary = 1 if (last_prev1 == 0 and first_prev2 == 1) else 0
        count_cur = (count_prev1 + count_prev2 + boundary) % MOD
        first_cur = first_prev1  # first char of F(i) = first char of F(i-1)
        last_cur = last_prev2  # last char of F(i) = last char of F(i-2)

        count_prev2 = count_prev1
        first_prev2 = first_prev1
        last_prev2 = last_prev1

        count_prev1 = count_cur
        first_prev1 = first_cur
        last_prev1 = last_cur

    return count_prev1
