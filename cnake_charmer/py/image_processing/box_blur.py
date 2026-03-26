"""Apply 3x3 box blur to an n x n grayscale image and return sum of blurred pixels.

Keywords: image processing, box blur, convolution, 2D, filter, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def box_blur(n: int) -> int:
    """Apply 3x3 box blur to an n x n grayscale image.

    Pixel[i][j] = (i*7 + j*13 + 3) % 256. The blur averages each pixel
    with its 8 neighbors (integer division). Border pixels are excluded
    from the output sum.

    Args:
        n: Image dimension (n x n).

    Returns:
        Sum of all blurred interior pixel values.
    """
    # Generate image as flat array
    img = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 7 + j * 13 + 3) % 256

    total = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            s = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    s += img[(i + di) * n + (j + dj)]
            total += s // 9

    return total
