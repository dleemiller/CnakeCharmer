"""
Apply a 3x3 Gaussian blur to an n*n image and return a checksum.

Keywords: image processing, gaussian blur, 2D, convolution, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def memview_gauss_blur_2d(n: int) -> float:
    """Apply 3x3 Gaussian blur on a deterministic n*n image, return checksum.

    Image pixels generated as: img[i][j] = ((i * 73 + j * 59 + 11) % 256).
    Gaussian kernel (unnormalized weights summing to 16):
        1 2 1
        2 4 2
        1 2 1

    Args:
        n: Dimension of the square image.

    Returns:
        Sum of all blurred pixels as a float.
    """
    # Build flat image
    img = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            img[i * n + j] = float((i * 73 + j * 59 + 11) % 256)

    # Gaussian kernel weights (3x3), sum = 16
    # 1 2 1 / 2 4 2 / 1 2 1
    kw = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]
    ksum = 16.0

    # Output image (only interior pixels)
    out = [0.0] * (n * n)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            val = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    val += img[(i + di) * n + (j + dj)] * kw[(di + 1) * 3 + (dj + 1)]
            out[i * n + j] = val / ksum

    # Checksum: sum of all output pixels
    checksum = 0.0
    for i in range(n * n):
        checksum += out[i]

    return checksum
