# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Find poly-A tail positions in deterministic DNA sequences (Cython-optimized).

Keywords: string processing, DNA, poly-A, tail finder, bioinformatics, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def poly_a_tail_finder(int n):
    """Generate n deterministic DNA sequences and find poly-A tail positions.

    Uses C-level integer arithmetic to determine bases and scan for tails
    without constructing actual string objects.
    """
    cdef int seq_len = 64
    cdef long long total_tail_lengths = 0
    cdef int count_with_tail = 0
    cdef int i, j, tail_len, base_idx, val
    cdef int seed

    for i in range(n):
        seed = (i * 7 + 13) % 1000003

        tail_len = 0
        for j in range(seq_len - 1, -1, -1):
            if j >= seq_len - 16:
                # Biased region
                val = (seed * (j + 1) + i * 3) % 7
                if val < 3:
                    base_idx = 0  # A
                elif val == 3:
                    base_idx = 4  # N
                else:
                    base_idx = (seed * j + i) % 4
            else:
                # Normal region
                base_idx = (seed * (j + 1) + i * 11) % 5

            # A=0, N=4 are tail bases
            if base_idx == 0 or base_idx == 4:
                tail_len = tail_len + 1
            else:
                break

        total_tail_lengths = total_tail_lengths + tail_len
        if tail_len > 0:
            count_with_tail = count_with_tail + 1

    return (total_tail_lengths, count_with_tail)
