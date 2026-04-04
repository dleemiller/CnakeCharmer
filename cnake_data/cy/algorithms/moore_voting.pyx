# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find majority element using Boyer-Moore voting algorithm.

Keywords: algorithms, moore voting, majority element, counting, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def moore_voting(int n):
    """Find majority element in arr[i] = (i*7+3) % 5, return element + count."""
    cdef int candidate = 0
    cdef int votes = 0
    cdef int val, i
    cdef int count = 0

    # Phase 1: find candidate
    for i in range(n):
        val = (i * 7 + 3) % 5
        if votes == 0:
            candidate = val
            votes = 1
        elif val == candidate:
            votes += 1
        else:
            votes -= 1

    # Phase 2: verify
    for i in range(n):
        val = (i * 7 + 3) % 5
        if val == candidate:
            count += 1

    return candidate + count
