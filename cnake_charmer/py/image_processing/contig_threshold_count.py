"""
Apply a threshold to 1D pixel data and count pixels above the threshold.

Keywords: image processing, threshold, binary, pixel, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def contig_threshold_count(n: int) -> int:
    """Apply threshold=128 to 1D pixel data, return count of pixels above threshold.

    Pixel data: pixels[i] = (i * 47 + 23) % 256.

    Args:
        n: Number of pixels.

    Returns:
        Count of pixels with value > 128.
    """
    threshold = 128
    count = 0
    for i in range(n):
        pixel = (i * 47 + 23) % 256
        if pixel > threshold:
            count += 1

    return count
