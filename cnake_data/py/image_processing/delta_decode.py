"""
Delta decoding for image compression (zip-with-prediction).

Keywords: image processing, delta decode, compression, prediction, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def delta_decode(n: int) -> tuple:
    """Decode an n×n delta-encoded image.

    Generate encoded data: encoded[y*n + x] = ((y*1009 + x*2003 + 42) * 17 + 137) & 0xFF
    Decode: decoded[y][x] = sum(encoded[y][0..x]) mod 256

    Args:
        n: Image width and height.

    Returns:
        (total_sum mod 2**32, decoded[n//2 * n + n//3])
    """
    size = n * n
    encoded = [0] * size
    decoded = [0] * size

    for y in range(n):
        for x in range(n):
            encoded[y * n + x] = ((y * 1009 + x * 2003 + 42) * 17 + 137) & 0xFF

    for y in range(n):
        cumsum = 0
        for x in range(n):
            cumsum = (cumsum + encoded[y * n + x]) & 0xFF
            decoded[y * n + x] = cumsum

    total_sum = sum(decoded) & 0xFFFFFFFF
    checksum = decoded[n // 2 * n + n // 3]
    return (total_sum, checksum)
