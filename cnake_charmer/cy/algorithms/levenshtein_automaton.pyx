# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute Levenshtein distances and count strings within threshold (Cython-optimized).

Keywords: levenshtein, edit distance, string matching, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def levenshtein_automaton(int n):
    """Count strings within edit distance 2 of HELLO using typed DP."""
    cdef int i, si, ti, cost, ins, dele, sub, val, row_min
    cdef int count = 0
    cdef int tlen = 5
    cdef int too_far
    cdef int prev[6]
    cdef int curr[6]
    cdef char target[5]
    cdef char chars[5]
    cdef long long v

    target[0] = 72  # H
    target[1] = 69  # E
    target[2] = 76  # L
    target[3] = 76  # L
    target[4] = 79  # O

    for i in range(n):
        v = <long long>i * 2654435761LL
        chars[0] = 65 + (v >> 0) % 16
        chars[1] = 65 + (v >> 4) % 16
        chars[2] = 65 + (v >> 8) % 16
        chars[3] = 65 + (v >> 12) % 16
        chars[4] = 65 + (v >> 16) % 16

        for ti in range(6):
            prev[ti] = ti

        too_far = 0
        for si in range(5):
            curr[0] = si + 1
            row_min = curr[0]
            for ti in range(tlen):
                if chars[si] == target[ti]:
                    cost = 0
                else:
                    cost = 1
                ins = curr[ti] + 1
                dele = prev[ti + 1] + 1
                sub = prev[ti] + cost
                val = ins
                if dele < val:
                    val = dele
                if sub < val:
                    val = sub
                curr[ti + 1] = val
                if val < row_min:
                    row_min = val
            if row_min > 2:
                too_far = 1
                break
            for ti in range(6):
                prev[ti] = curr[ti]

        if too_far == 0 and prev[tlen] <= 2:
            count += 1

    return count
