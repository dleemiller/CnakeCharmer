"""Apply gamma correction to an n x n deterministic grayscale image.

Keywords: image processing, gamma correction, brightness, power law, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def gamma_correction(n: int) -> tuple:
    """Apply gamma correction to an n x n image with three different gamma values.

    Pixel[i][j] = (i*17 + j*31) % 256.
    Gamma correction: out = 255 * (in / 255) ^ gamma, rounded to int, clamped [0, 255].
    Applies gamma = 0.5, 1.5, and 2.2 and returns sums for each.

    Args:
        n: Image dimension (n x n).

    Returns:
        Tuple of (sum_gamma_05, sum_gamma_15, sum_gamma_22).
    """
    gammas = [0.5, 1.5, 2.2]
    inv_255 = 1.0 / 255.0

    # Build lookup tables for each gamma (256 entries each)
    luts = [[0] * 256, [0] * 256, [0] * 256]
    for g_idx in range(3):
        gamma = gammas[g_idx]
        for v in range(256):
            corrected = 255.0 * math.pow(v * inv_255, gamma)
            rounded = int(corrected + 0.5)
            if rounded < 0:
                rounded = 0
            elif rounded > 255:
                rounded = 255
            luts[g_idx][v] = rounded

    # Generate image and apply each gamma LUT
    sums = [0, 0, 0]
    for i in range(n):
        for j in range(n):
            pixel = (i * 17 + j * 31) % 256
            for g_idx in range(3):
                sums[g_idx] += luts[g_idx][pixel]

    return (sums[0], sums[1], sums[2])
