"""Otsu's method for automatic image thresholding.

Computes the optimal threshold that minimizes intra-class variance
for a deterministic grayscale image.

Keywords: image processing, Otsu, threshold, segmentation, histogram, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(800,))
def otsu_threshold(n: int) -> tuple:
    """Compute Otsu threshold for an n x n deterministic grayscale image.

    Pixel[i][j] = (i*7 + j*13 + i*j*3 + 5) % 256.

    Args:
        n: Image dimension (n x n).

    Returns:
        Tuple of (threshold, foreground_count, variance_between).
    """
    total_pixels = n * n

    # Build histogram
    hist = [0] * 256
    for i in range(n):
        for j in range(n):
            val = (i * 7 + j * 13 + i * j * 3 + 5) % 256
            hist[val] += 1

    # Compute total mean
    total_sum = 0.0
    for t in range(256):
        total_sum += t * hist[t]

    best_threshold = 0
    best_variance = 0.0

    w0 = 0
    sum0 = 0.0

    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total_pixels - w0
        if w1 == 0:
            break

        sum0 += t * hist[t]
        sum1 = total_sum - sum0

        mean0 = sum0 / w0
        mean1 = sum1 / w1

        variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1)

        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    # Count foreground pixels (above threshold)
    foreground_count = 0
    for t in range(best_threshold + 1, 256):
        foreground_count += hist[t]

    return (best_threshold, foreground_count, best_variance)
