"""Compute histogram equalization on an n x n grayscale image.

Keywords: image processing, histogram equalization, contrast, CDF, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def histogram_equalize(n: int) -> int:
    """Perform histogram equalization on an n x n image and return sum of equalized values.

    Pixel[i][j] = (i*17 + j*31 + 5) % 256. Computes the histogram, then the
    cumulative distribution function, and maps each pixel to its equalized
    value: equalized = round(CDF(pixel) * 255 / total_pixels).

    Args:
        n: Image dimension (n x n).

    Returns:
        Sum of all equalized pixel values.
    """
    size = n * n

    # Generate image as flat array
    img = [0] * size
    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 17 + j * 31 + 5) % 256

    # Compute histogram
    hist = [0] * 256
    for i in range(size):
        hist[img[i]] += 1

    # Compute CDF
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Find min non-zero CDF value
    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break

    # Equalize and sum
    total = 0
    denom = size - cdf_min
    if denom == 0:
        denom = 1
    for i in range(size):
        equalized = ((cdf[img[i]] - cdf_min) * 255 + denom // 2) // denom
        total += equalized

    return total
