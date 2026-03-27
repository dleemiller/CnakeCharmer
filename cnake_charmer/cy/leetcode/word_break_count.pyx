# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count ways to segment a string into dictionary words (Cython-optimized).

Keywords: leetcode, word break, dynamic programming, segmentation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark

# Max encoded value: sum(4 * 5^k for k in 0..5) = 4*(5^6-1)/(5-1) = 3906*4 = 15624
# So dict_arr needs 15625 entries (5^6).
DEF DICT_ARR_SIZE = 15625


@cython_benchmark(syntax="cy", args=(3000,))
def word_break_count(int n):
    """Count word break segmentations using integer-encoded C array dictionary."""
    cdef long long mod = 1000000007

    cdef int *s = <int *>malloc(n * sizeof(int))
    if not s:
        raise MemoryError()

    cdef int i, length, k
    for i in range(n):
        s[i] = (i * 7 + 3) % 5

    # Build dictionary: encode each word as a base-5 integer
    cdef int prefix_len = 100 if n >= 100 else n
    cdef int max_word = 6

    # C array of booleans indexed by encoded word value
    cdef int *dict_arr = <int *>malloc(DICT_ARR_SIZE * sizeof(int))
    if not dict_arr:
        free(s)
        raise MemoryError()
    memset(dict_arr, 0, DICT_ARR_SIZE * sizeof(int))

    cdef int encoded, power
    for i in range(prefix_len):
        for length in range(1, max_word + 1):
            if i + length <= prefix_len:
                encoded = 0
                power = 1
                for k in range(length):
                    encoded += s[i + k] * power
                    power *= 5
                dict_arr[encoded] = 1

    # DP array (C int array, values kept mod 1e9+7 so int is sufficient)
    cdef long long *dp = <long long *>malloc((n + 1) * sizeof(long long))
    if not dp:
        free(s)
        free(dict_arr)
        raise MemoryError()
    memset(dp, 0, (n + 1) * sizeof(long long))
    dp[0] = 1

    cdef int max_used = 0

    with nogil:
        for i in range(1, n + 1):
            for length in range(1, max_word + 1):
                if length > i:
                    break
                # Encode s[i-length : i] in base-5
                encoded = 0
                power = 1
                for k in range(length):
                    encoded += s[i - length + k] * power
                    power *= 5
                if dict_arr[encoded]:
                    dp[i] = (dp[i] + dp[i - length]) % mod
                    if dp[i - length] > 0 and length > max_used:
                        max_used = length

    cdef int num_nonzero = 0
    for i in range(n + 1):
        if dp[i] > 0:
            num_nonzero += 1

    cdef long long result = dp[n]
    free(s)
    free(dict_arr)
    free(dp)

    return (int(result), max_used, num_nonzero)
