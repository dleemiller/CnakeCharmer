"""Pack/unpack pixel values using bitwise ops and compute channel averages.

Keywords: union, color, channels, image processing, bitwise, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def union_color_channels(n: int) -> int:
    """Pack n pixel values, extract channels, compute averages.

    Uses bitwise operations to pack/unpack RGBA channels from a
    32-bit unsigned integer, emulating a C union approach.

    Args:
        n: Number of pixels to process.

    Returns:
        r_avg * 1000 + g_avg as int.
    """
    r_sum = 0
    g_sum = 0
    for i in range(n):
        # Deterministic hash to get a packed pixel
        packed = (i * 2654435761) & 0xFFFFFFFF
        # Extract channels via bitwise ops (little-endian byte order)
        r = packed & 0xFF
        g = (packed >> 8) & 0xFF
        r_sum += r
        g_sum += g
    r_avg = r_sum // n
    g_avg = g_sum // n
    return r_avg * 1000 + g_avg
