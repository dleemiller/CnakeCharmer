# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count characters and compute frequency-rank weighted sum (Cython-optimized).

Keywords: sorting, counting sort, character frequency, ranking, cython, benchmark
"""

from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def counting_sort_chars(int n):
    """Sort characters by frequency and compute weighted rank sum."""
    cdef int counts[26]
    memset(counts, 0, 26 * sizeof(int))

    cdef int i, j, temp

    # Count character frequencies
    for i in range(n):
        counts[(i * 7 + 3) % 26] += 1

    # Collect non-zero frequencies into a sortable array
    cdef int freq_list[26]
    cdef int num_freqs = 0
    for i in range(26):
        if counts[i] > 0:
            freq_list[num_freqs] = counts[i]
            num_freqs += 1

    # Sort frequencies descending (insertion sort, small array)
    for i in range(1, num_freqs):
        temp = freq_list[i]
        j = i
        while j > 0 and freq_list[j - 1] < temp:
            freq_list[j] = freq_list[j - 1]
            j -= 1
        freq_list[j] = temp

    # Compute sum of freq * rank (1-indexed)
    cdef long long total = 0
    for i in range(num_freqs):
        total += <long long>freq_list[i] * (i + 1)

    return int(total)
