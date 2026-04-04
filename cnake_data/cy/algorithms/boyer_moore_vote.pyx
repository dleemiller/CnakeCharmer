# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Heavy hitter detection using Boyer-Moore majority vote variant (Cython-optimized).

Keywords: algorithms, boyer-moore, majority vote, heavy hitter, frequency, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def boyer_moore_vote(int n):
    """Detect top-2 heavy hitters using Boyer-Moore majority vote."""
    cdef int cand1 = 0, cand2 = 0
    cdef int count1 = 0, count2 = 0
    cdef int i, val
    cdef int freq1 = 0, freq2 = 0
    cdef int tmp

    # Phase 1: Find two candidates
    for i in range(n):
        val = ((i * 17 + 3) ^ (i * 31 + 7)) % 50

        if val == cand1:
            count1 += 1
        elif val == cand2:
            count2 += 1
        elif count1 == 0:
            cand1 = val
            count1 = 1
        elif count2 == 0:
            cand2 = val
            count2 = 1
        else:
            count1 -= 1
            count2 -= 1

    # Phase 2: Count actual frequencies
    for i in range(n):
        val = ((i * 17 + 3) ^ (i * 31 + 7)) % 50
        if val == cand1:
            freq1 += 1
        elif val == cand2:
            freq2 += 1

    # Ensure cand1 has higher frequency
    if freq2 > freq1:
        tmp = cand1; cand1 = cand2; cand2 = tmp
        tmp = freq1; freq1 = freq2; freq2 = tmp

    return (cand1, freq1, cand2)
