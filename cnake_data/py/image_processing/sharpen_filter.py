"""Apply a 3x3 sharpening (unsharp mask) filter to a deterministic image.

Keywords: image processing, sharpen, unsharp mask, convolution, kernel, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def sharpen_filter(n: int) -> tuple:
    """Apply a 3x3 sharpening kernel to an n x n image, return statistics.

    Pixel[i][j] = (i*17 + j*31) % 256.
    The sharpening kernel is:
        [ 0, -1,  0]
        [-1,  5, -1]
        [ 0, -1,  0]
    Output pixels are clamped to [0, 255].

    Args:
        n: Image dimension (n x n).

    Returns:
        Tuple of (sum_of_output_pixels, count_of_clipped_pixels).
    """
    # Sharpening kernel (3x3)
    kernel = [
        0,
        -1,
        0,
        -1,
        5,
        -1,
        0,
        -1,
        0,
    ]

    # Generate image as flat list
    image = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            image[i * n + j] = (i * 17 + j * 31) % 256

    # Apply sharpening (skip 1-pixel border)
    total = 0
    clipped = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            acc = 0
            for ki in range(3):
                for kj in range(3):
                    acc += kernel[ki * 3 + kj] * image[(i + ki - 1) * n + (j + kj - 1)]
            # Clamp to [0, 255]
            if acc < 0:
                acc = 0
                clipped += 1
            elif acc > 255:
                acc = 255
                clipped += 1
            total += acc

    return (total, clipped)
