"""Sigma (range) filter on a 2D floating-point image.

For each pixel, averages all neighbors within a square window whose value
is within `threshold` of the center pixel. Similar to a bilateral filter
but with a hard threshold instead of a Gaussian kernel.

Keywords: image processing, sigma filter, bilateral, smoothing, neighborhood, 2d
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(60, 80, 3, 0.5))
def sigma_filter_2d(n: int, m: int, radius: int, threshold: float) -> tuple:
    """Apply sigma filter to a deterministically generated n×m image.

    Args:
        n: Image height (rows).
        m: Image width (columns).
        radius: Half-window size (window is (2*radius+1)²).
        threshold: Pixel inclusion threshold.

    Returns:
        Tuple of (top_left_pixel, center_pixel, quarter_pixel).
    """
    # Generate deterministic source image in [-1, 1]
    src = [[math.sin(i * 0.5 + j * 0.7) for j in range(m)] for i in range(n)]
    dst = [[0.0] * m for _ in range(n)]

    for y in range(n):
        for x in range(m):
            center = src[y][x]
            acc = 0.0
            count = 0
            for dj in range(-radius, radius + 1):
                yy = max(0, min(y + dj, n - 1))
                for di in range(-radius, radius + 1):
                    xx = max(0, min(x + di, m - 1))
                    val = src[yy][xx]
                    if abs(center - val) < threshold:
                        acc += val
                        count += 1
            dst[y][x] = acc / count if count > 0 else 0.0

    return (dst[0][0], dst[n // 2][m // 2], dst[n // 4][m // 4])
