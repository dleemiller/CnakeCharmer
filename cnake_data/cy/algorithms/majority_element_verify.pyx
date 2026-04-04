# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find and verify majority elements in multiple array segments using Boyer-Moore.

Keywords: algorithms, majority element, boyer moore, verification, voting, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def majority_element_verify(int n):
    """Find majority elements across multiple segments and verify them."""
    cdef int seg_size = 1
    while (seg_size + 1) * (seg_size + 1) <= n:
        seg_size += 1

    cdef int verified_count = 0
    cdef int majority_sum = 0
    cdef int no_majority_count = 0
    cdef int seg_start = 0
    cdef int seg_end, seg_len
    cdef int candidate, votes, count, val, i

    while seg_start < n:
        seg_end = seg_start + seg_size
        if seg_end > n:
            seg_end = n
        seg_len = seg_end - seg_start

        # Phase 1: Boyer-Moore voting
        candidate = 0
        votes = 0
        for i in range(seg_start, seg_end):
            val = (i * 41 + 7) % 17
            if votes == 0:
                candidate = val
                votes = 1
            elif val == candidate:
                votes += 1
            else:
                votes -= 1

        # Phase 2: verification
        count = 0
        for i in range(seg_start, seg_end):
            val = (i * 41 + 7) % 17
            if val == candidate:
                count += 1

        if count * 2 > seg_len:
            verified_count += 1
            majority_sum += candidate
        else:
            no_majority_count += 1

        seg_start = seg_end

    return (verified_count, majority_sum, no_majority_count)
