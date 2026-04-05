"""Compare tile/sample grids by squared-difference argmin selection.

Adapted from The Stack v2 Cython candidate:
- blob_id: a356283380de9e9e36e19f6920684ad6180dc655
- filename: matrix_math.pyx

Keywords: algorithms, matrix, tile matching, argmin, squared difference
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(20, 20, 28, 24))
def stack2_tile_min_diffs(
    tile_rows: int, tile_cols: int, tile_count: int, sample_count: int
) -> tuple:
    """Find best matching tile index for each sample tile using SSD metric."""
    pixel_count = tile_rows * tile_cols
    best_sum = 0
    first_best = -1
    checksum = 0

    for sample_idx in range(sample_count):
        best_score = 1 << 62
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
        best_sum += best_score
        checksum = (checksum + (best_tile + 5) * (sample_idx + 13)) & 0xFFFFFFFF

    return (sample_count, first_best, best_sum & 0xFFFFFFFF, checksum)
