"""Flip an image horizontally and vertically, compute checksum.

Keywords: image, flip, transform, checksum, image processing, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def image_flip_checksum(n: int) -> int:
    """Generate an n x n grayscale image, flip it horizontally then vertically,
    and compute a weighted checksum.

    Args:
        n: Image dimension (n x n pixels).

    Returns:
        Weighted checksum of the double-flipped image.
    """
    # Generate image (flat row-major)
    img = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            h = ((i * 2654435761 + j * 1103515245 + 7) >> 4) & 0xFF
            img[i * n + j] = h

    # Horizontal flip (mirror left-right)
    flipped = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            flipped[i * n + j] = img[i * n + (n - 1 - j)]

    # Vertical flip (mirror top-bottom)
    result = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            result[i * n + j] = flipped[(n - 1 - i) * n + j]

    # Weighted checksum
    checksum = 0
    for i in range(n * n):
        checksum += result[i] * ((i % 256) + 1)
        checksum &= 0x7FFFFFFF

    return checksum
