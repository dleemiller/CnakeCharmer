"""Binary morphological dilation with 3x3 kernel.

Keywords: image processing, morphological dilation, binary image, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def dilation(n: int) -> int:
    """Apply binary morphological dilation with 3x3 kernel on n x n image.

    Pixel[i][j] = 1 if (i*7 + j*13 + 3) % 10 == 0 else 0.
    A pixel becomes 1 if any of its 3x3 neighbors (including itself) is 1.

    Args:
        n: Image dimension (n x n).

    Returns:
        Count of 1-pixels after dilation.
    """
    # Build flat image
    img = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13 + 3) % 10 == 0:
                img[i * n + j] = 1

    # Dilate: pixel (i,j) is 1 if any 3x3 neighbor is 1
    count = 0
    for i in range(n):
        for j in range(n):
            dilated = 0
            i_start = i - 1 if i > 0 else 0
            i_end = i + 1 if i < n - 1 else n - 1
            j_start = j - 1 if j > 0 else 0
            j_end = j + 1 if j < n - 1 else n - 1
            for di in range(i_start, i_end + 1):
                for dj in range(j_start, j_end + 1):
                    if img[di * n + dj] == 1:
                        dilated = 1
                        break
                if dilated == 1:
                    break
            count += dilated

    return count
