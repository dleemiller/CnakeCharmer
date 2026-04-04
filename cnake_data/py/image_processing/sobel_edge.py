"""Apply Sobel edge detection on an n x n grayscale image.

Keywords: image processing, Sobel, edge detection, gradient, convolution, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def sobel_edge(n: int) -> int:
    """Apply Sobel edge detection and count pixels with gradient magnitude above threshold.

    Pixel[i][j] = (i*7 + j*13 + 3) % 256. Computes horizontal and vertical
    Sobel gradients, then gradient magnitude = sqrt(gx^2 + gy^2). Counts
    interior pixels where magnitude > 100.

    Args:
        n: Image dimension (n x n).

    Returns:
        Count of edge pixels (gradient magnitude > 100).
    """
    # Generate image as flat array
    img = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 7 + j * 13 + 3) % 256

    # Sobel kernels
    # Gx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Gy: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    count = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            gx = (
                -img[(i - 1) * n + (j - 1)]
                + img[(i - 1) * n + (j + 1)]
                - 2 * img[i * n + (j - 1)]
                + 2 * img[i * n + (j + 1)]
                - img[(i + 1) * n + (j - 1)]
                + img[(i + 1) * n + (j + 1)]
            )

            gy = (
                -img[(i - 1) * n + (j - 1)]
                - 2 * img[(i - 1) * n + j]
                - img[(i - 1) * n + (j + 1)]
                + img[(i + 1) * n + (j - 1)]
                + 2 * img[(i + 1) * n + j]
                + img[(i + 1) * n + (j + 1)]
            )

            mag = math.sqrt(gx * gx + gy * gy)
            if mag > 100.0:
                count += 1

    return count
