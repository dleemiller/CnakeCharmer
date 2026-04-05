# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply triangular masking modes to deterministic square matrices (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: e58e6e13acdc9a91efb97324f2487b4a058a4506
- filename: triangle.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(260, 2, 41))
def stack2_triangle_modes(int side_len, int mode_code, int seed_base):
    cdef unsigned int state = <unsigned int>((246813579 + seed_base * 11939) & 0xFFFFFFFF)
    cdef int total_cells = side_len * side_len
    cdef int *matrix = <int *>malloc(total_cells * sizeof(int))
    cdef int row, col, idx, value
    cdef long long diag_sum = 0
    cdef int nonzero_count = 0
    cdef unsigned int checksum = 0
    cdef int corner = 0

    if not matrix:
        raise MemoryError()

    for row in range(side_len):
        for col in range(side_len):
            state = (1664525 * state + 1013904223)
            idx = row * side_len + col
            matrix[idx] = <int>((state >> 11) % 2048) - 1024

    for row in range(side_len):
        for col in range(side_len):
            idx = row * side_len + col
            value = matrix[idx]
            if mode_code == 0:
                if row < col:
                    value = 0
            elif mode_code == 1:
                if row <= col:
                    value = 0
            elif mode_code == 2:
                if row == col:
                    value = 1
                elif row < col:
                    value = 0
            else:
                if row > col:
                    value = 0

            if row == col:
                diag_sum += value
            if value != 0:
                nonzero_count += 1
            checksum = (checksum + <unsigned int>((value & 0xFFFF) * (row + 3) * (col + 7))) & 0xFFFFFFFF
            if row == side_len - 1 and col == side_len - 1:
                corner = value

    free(matrix)
    return (diag_sum, nonzero_count, checksum, corner)
