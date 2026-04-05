"""Encode/decode compact columnar bitfields with packed metadata.

Adapted from The Stack v2 Cython candidate:
- blob_id: 8934ca1b0452cba1cf6d0ab21e8e79b9811176d9
- filename: __functions_cy.pyx

Keywords: algorithms, bit packing, encoding, decoding, columnar
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(8, 8, 19, 0, 4000))
def stack2_bitfield_codec(
    col_count: int, row_count: int, seed_tag: int, trim_bits: int, repeat_count: int = 1
) -> tuple:
    """Pack board-like columns into u64 and decode summary values."""
    total_bits = col_count * row_count + col_count * trim_bits
    if total_bits <= 0 or total_bits > 64:
        return (0, 0, 0, 0)

    mask64 = 0xFFFFFFFFFFFFFFFF
    summary_encoded = 0
    summary_ones = 0
    summary_height = 0
    summary_parity = 0

    for rep_idx in range(repeat_count):
        state = (123123123 + (seed_tag + rep_idx) * 10007) & mask64
        encoded = 0
        free_lens = [0] * col_count
        for col in range(col_count):
            state = (1664525 * state + 1013904223) & mask64
            free_len = (state >> 9) % (row_count + 1)
            free_lens[col] = free_len
            fill_cells = row_count - free_len

            for _ in range(fill_cells):
                state = (1664525 * state + 1013904223) & mask64
                bit = (state >> 15) & 1
                encoded = ((encoded << 1) | bit) & mask64
            for _ in range(free_len):
                encoded = (encoded << 1) & mask64

        for free_len in free_lens:
            val = free_len
            for bit_pos in range(trim_bits - 1, -1, -1):
                encoded = ((encoded << 1) | ((val >> bit_pos) & 1)) & mask64

        ones_count = 0
        parity = 0
        height_checksum = 0
        payload_size = col_count * row_count
        for col in range(col_count):
            meta_shift = col_count * trim_bits - (col + 1) * trim_bits
            free_len = (encoded >> meta_shift) & ((1 << trim_bits) - 1)
            fill_cells = row_count - free_len
            height_checksum += (col + 1) * fill_cells
            col_start = payload_size - (col + 1) * row_count
            for row in range(fill_cells):
                bit = (encoded >> (col_start + (row_count - 1 - row))) & 1
                ones_count += bit
                parity ^= bit

        summary_encoded = encoded
        summary_ones = (summary_ones + ones_count) & 0xFFFFFFFF
        summary_height = (summary_height + height_checksum) & 0xFFFFFFFF
        summary_parity ^= parity

    return (summary_encoded, summary_ones, summary_height, summary_parity)
