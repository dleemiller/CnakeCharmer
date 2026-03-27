# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count ways to segment a string into dictionary words (Cython-optimized).

Keywords: leetcode, word break, dynamic programming, segmentation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def word_break_count(int n):
    """Count word break segmentations using typed DP array."""
    cdef long long mod = 1000000007

    cdef int *s = <int *>malloc(n * sizeof(int))
    if not s:
        raise MemoryError()

    cdef int i, length
    for i in range(n):
        s[i] = (i * 7 + 3) % 5

    # Build dictionary from first min(100, n) chars
    cdef int prefix_len = 100 if n >= 100 else n
    cdef int max_word = 6

    dictionary = set()
    for i in range(prefix_len):
        for length in range(1, max_word + 1):
            if i + length <= prefix_len:
                word = tuple(s[i + k] for k in range(length))
                dictionary.add(word)

    # DP array
    cdef long long *dp = <long long *>malloc((n + 1) * sizeof(long long))
    if not dp:
        free(s)
        raise MemoryError()
    memset(dp, 0, (n + 1) * sizeof(long long))
    dp[0] = 1

    cdef int max_used = 0
    cdef int k

    for i in range(1, n + 1):
        for length in range(1, max_word + 1):
            if length > i:
                break
            word = tuple(s[i - length + k] for k in range(length))
            if word in dictionary:
                dp[i] = (dp[i] + dp[i - length]) % mod
                if dp[i - length] > 0 and length > max_used:
                    max_used = length

    cdef int num_nonzero = 0
    for i in range(n + 1):
        if dp[i] > 0:
            num_nonzero += 1

    cdef long long result = dp[n]
    free(s)
    free(dp)

    return (int(result), max_used, num_nonzero)
