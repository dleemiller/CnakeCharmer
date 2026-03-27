"""Byte histogram via buffer-backed unsigned char array.

Fills a bytearray with hash-derived values, builds a 256-bin
histogram, and returns the maximum bin count.

Keywords: algorithms, buffer protocol, histogram, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def buffer_byte_histogram(n: int) -> int:
    """Build byte histogram and return max bin count.

    Args:
        n: Number of bytes to process.

    Returns:
        Maximum frequency among the 256 bins.
    """
    mask = 0xFFFFFFFF
    data = bytearray(n)
    for i in range(n):
        h = ((i * 2654435761) & mask) ^ ((i * 2246822519) & mask)
        data[i] = (h >> 8) & 0xFF

    histogram = [0] * 256
    for i in range(n):
        histogram[data[i]] += 1

    max_count = 0
    for i in range(256):
        if histogram[i] > max_count:
            max_count = histogram[i]
    return max_count
