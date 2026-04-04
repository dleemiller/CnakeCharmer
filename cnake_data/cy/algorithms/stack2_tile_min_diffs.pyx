# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compare tile/sample grids by squared-difference argmin selection (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: a356283380de9e9e36e19f6920684ad6180dc655
- filename: matrix_math.pyx
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(20, 20, 28, 24))
def stack2_tile_min_diffs(int tile_rows, int tile_cols, int tile_count, int sample_count):
    cdef int pixel_count = tile_rows * tile_cols
    cdef int sample_idx, tile_idx, pixel_idx
    cdef int left, right, diff
    cdef long long best_score, score
    cdef int best_tile, first_best = -1
    cdef unsigned int checksum = 0
    cdef unsigned int best_sum = 0

    for sample_idx in range(sample_count):
        best_score = 9223372036854775807
        best_tile = 0
        for tile_idx in range(tile_count):
            score = 0
            for pixel_idx in range(pixel_count):
                left = (tile_idx * 97 + pixel_idx * 31 + 19) % 256
                right = (sample_idx * 89 + pixel_idx * 37 + 7) % 256
                diff = left - right
                score += diff * diff
            if score < best_score:
                best_score = score
                best_tile = tile_idx

        if sample_idx == 0:
            first_best = best_tile
        best_sum = (best_sum + <unsigned int>best_score) & 0xFFFFFFFF
        checksum = (checksum + <unsigned int>((best_tile + 5) * (sample_idx + 13))) & 0xFFFFFFFF

    return (sample_count, first_best, best_sum, checksum)
