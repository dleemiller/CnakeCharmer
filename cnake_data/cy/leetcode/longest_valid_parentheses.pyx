# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find longest valid parentheses substring using stack-based DP (Cython-optimized).

Keywords: leetcode, parentheses, longest, valid, stack, dynamic programming, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def longest_valid_parentheses(int n):
    """Find longest valid parentheses substring in a deterministic string of length n."""
    cdef int i, sp, max_length, start_pos, total_valid, in_valid
    cdef int *s = <int *>malloc(n * sizeof(int))
    cdef int *dp = <int *>malloc(n * sizeof(int))
    cdef int *stack = <int *>malloc((n + 1) * sizeof(int))

    if not s or not dp or not stack:
        if s: free(s)
        if dp: free(dp)
        if stack: free(stack)
        raise MemoryError()

    cdef long long lcg = 123456789
    for i in range(n):
        lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
        s[i] = (lcg >> 16) & 1

    for i in range(n):
        dp[i] = 0

    stack[0] = -1
    sp = 1

    for i in range(n):
        if s[i] == 0:
            stack[sp] = i
            sp = sp + 1
        else:
            sp = sp - 1
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

    total_valid = 0
    in_valid = 0
    for i in range(n):
        if dp[i] > 0:
            if in_valid == 0:
                total_valid = total_valid + 1
                in_valid = 1
        else:
            in_valid = 0

    free(s)
    free(dp)
    free(stack)

    return (max_length, start_pos, total_valid)
