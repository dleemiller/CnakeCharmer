"""Apply triangular masking modes to deterministic square matrices.

Adapted from The Stack v2 Cython candidate:
- blob_id: e58e6e13acdc9a91efb97324f2487b4a058a4506
- filename: triangle.pyx

Keywords: numerical, matrix, triangle, masking, checksums
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(260, 2, 41))
def stack2_triangle_modes(side_len: int, mode_code: int, seed_base: int) -> tuple:
    """Mask matrix entries according to mode and return summary tuple."""
    state = (246813579 + seed_base * 11939) & 0xFFFFFFFF
    matrix = [[0] * side_len for _ in range(side_len)]

    for row in range(side_len):
        for col in range(side_len):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            matrix[row][col] = ((state >> 11) % 2048) - 1024

    diag_sum = 0
    nonzero_count = 0
    checksum = 0
    corner = 0

    for row in range(side_len):
        for col in range(side_len):
            value = matrix[row][col]
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
            checksum = (checksum + (value & 0xFFFF) * (row + 3) * (col + 7)) & 0xFFFFFFFF
            if row == side_len - 1 and col == side_len - 1:
                corner = value

    return (diag_sum, nonzero_count, checksum, corner)
