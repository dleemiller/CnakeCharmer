"""2D convolution with edge detection kernel.

Keywords: convolution, 2d, edge detection, image, neural network
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def conv2d(n: int) -> int:
    """Convolve n x n image with 3x3 edge detection kernel.

    Pixel[i][j] = (i*17 + j*31 + 5) % 256.
    Kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]].

    Args:
        n: Image dimension (n x n).

    Returns:
        Sum of convolution output.
    """
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    total = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            s = 0
            for ki in range(3):
                for kj in range(3):
                    pixel = ((i - 1 + ki) * 17 + (j - 1 + kj) * 31 + 5) % 256
                    s += pixel * kernel[ki][kj]
            total += s
    return total
