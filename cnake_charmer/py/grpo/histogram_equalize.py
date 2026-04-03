"""Histogram equalization on grayscale pixel data.

Keywords: grpo, image processing, histogram, equalization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def histogram_equalize(n: int) -> tuple:
    """Apply histogram equalization to deterministic grayscale pixel data.

    Maps pixel intensities [0-255] through a CDF-based transfer function
    to spread the histogram evenly.

    Returns (mean output value, count of pixels at intensity 128, checksum).

    Args:
        n: Number of pixels.

    Returns:
        Tuple of (mean_value, count_128, checksum).
    """
    # Generate deterministic pixel data (biased toward darks)
    pixels = [0] * n
    for i in range(n):
        # Skewed distribution: lots of low values
        v = ((i * 73 + 17) % 256 * (i * 31 + 3) % 256) % 256
        pixels[i] = v

    # Build histogram
    hist = [0] * 256
    for p in pixels:
        hist[p] += 1

    # Build CDF
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Find min CDF (first non-zero)
    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break

    # Build transfer function
    denom = n - cdf_min
    if denom <= 0:
        denom = 1
    transfer = [0] * 256
    for i in range(256):
        if cdf[i] > 0:
            transfer[i] = round((cdf[i] - cdf_min) * 255.0 / denom)
        else:
            transfer[i] = 0

    # Apply transfer
    total = 0
    count_128 = 0
    checksum = 0
    for i in range(n):
        out = transfer[pixels[i]]
        total += out
        if out == 128:
            count_128 += 1
        checksum = (checksum + out * (i & 0xFF)) & 0xFFFFFFFF

    mean_val = round(total / n, 2) if n > 0 else 0.0

    return (mean_val, count_128, checksum)
