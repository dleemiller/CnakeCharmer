"""Apply 5x5 Gaussian blur to a deterministic image.

Keywords: image processing, gaussian blur, convolution, kernel, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(400,))
def gaussian_blur(n: int) -> int:
    """Apply 5x5 Gaussian blur to an n x n image, return sum of blurred pixels.

    Pixel[i][j] = (i*7 + j*13 + 3) % 256.
    Uses a 5x5 Gaussian kernel (sigma~1).

    Args:
        n: Image dimension (n x n).

    Returns:
        Sum of all blurred pixel values (as integers).
    """
    # 5x5 Gaussian kernel (unnormalized, sigma~1)
    kernel = [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ]
    kernel_sum = 273  # sum of all kernel values

    # Generate image as flat list
    image = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            image[i * n + j] = (i * 7 + j * 13 + 3) % 256

    # Apply blur (skip 2-pixel border)
    total = 0
    for i in range(2, n - 2):
        for j in range(2, n - 2):
            acc = 0
            for ki in range(5):
                for kj in range(5):
                    acc += kernel[ki][kj] * image[(i + ki - 2) * n + (j + kj - 2)]
            total += acc // kernel_sum

    return total
