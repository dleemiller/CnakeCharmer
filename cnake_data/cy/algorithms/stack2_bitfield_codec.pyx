# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Encode/decode compact columnar bitfields with packed metadata (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 8934ca1b0452cba1cf6d0ab21e8e79b9811176d9
- filename: __functions_cy.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(8, 8, 19, 0, 4000))
def stack2_bitfield_codec(
    int col_count, int row_count, int seed_tag, int trim_bits, int repeat_count=1
):
    cdef int total_bits = col_count * row_count + col_count * trim_bits
    cdef unsigned long long mask64 = 0xFFFFFFFFFFFFFFFF
    cdef unsigned long long state
    cdef unsigned long long encoded = 0
    cdef int *free_lens
    cdef int col, row, bit_pos
    cdef int free_len, fill_cells
    cdef unsigned long long bit
    cdef int ones_count = 0
    cdef int parity = 0
    cdef int height_checksum = 0
    cdef int payload_bits
    cdef int meta_shift, col_start
    cdef int rep_idx
    cdef unsigned int summary_ones = 0
    cdef unsigned int summary_height = 0
    cdef int summary_parity = 0

    if total_bits <= 0 or total_bits > 64:
        return (0, 0, 0, 0)

    free_lens = <int *>malloc(col_count * sizeof(int))
    if not free_lens:
        raise MemoryError()

    for rep_idx in range(repeat_count):
        state = <unsigned long long>(123123123 + (seed_tag + rep_idx) * 10007)
        encoded = 0
        ones_count = 0
        parity = 0
        height_checksum = 0

        for col in range(col_count):
            state = (1664525 * state + 1013904223) & mask64
            free_len = <int>((state >> 9) % <unsigned long long>(row_count + 1))
            free_lens[col] = free_len
            fill_cells = row_count - free_len

            for row in range(fill_cells):
                state = (1664525 * state + 1013904223) & mask64
                bit = (state >> 15) & 1
                encoded = ((encoded << 1) | bit) & mask64
            for row in range(free_len):
                encoded = (encoded << 1) & mask64

        for col in range(col_count):
            free_len = free_lens[col]
            for bit_pos in range(trim_bits - 1, -1, -1):
                encoded = ((encoded << 1) | ((free_len >> bit_pos) & 1)) & mask64

        payload_bits = col_count * row_count
        for col in range(col_count):
            meta_shift = col_count * trim_bits - (col + 1) * trim_bits
            free_len = <int>((encoded >> meta_shift) & ((1 << trim_bits) - 1))
            fill_cells = row_count - free_len
            height_checksum += (col + 1) * fill_cells

            col_start = payload_bits - (col + 1) * row_count
            for row in range(fill_cells):
                bit = (encoded >> (col_start + (row_count - 1 - row))) & 1
                ones_count += <int>bit
                parity ^= <int>bit

        summary_ones = (summary_ones + <unsigned int>ones_count) & 0xFFFFFFFF
        summary_height = (summary_height + <unsigned int>height_checksum) & 0xFFFFFFFF
        summary_parity ^= parity

    free(free_lens)
    return (encoded, summary_ones, summary_height, summary_parity)
