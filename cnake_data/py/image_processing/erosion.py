"""Binary morphological erosion with 3x3 kernel.

Keywords: image processing, morphological erosion, binary image, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def erosion(n: int) -> int:
    """Apply binary morphological erosion with 3x3 kernel on n x n image.

    Pixel[i][j] = 1 if (i*i + j*j) % 17 < 14 else 0.
    A pixel survives erosion only if all 9 neighbors (including itself)
    in the 3x3 window are 1.

    Args:
        n: Image dimension (n x n).

    Returns:
        Count of remaining 1-pixels after erosion.
    """
    # Build flat image
    img = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            if (i * i + j * j) % 17 < 14:
                img[i * n + j] = 1

    # Erode: pixel (i,j) is 1 only if all 3x3 neighbors are 1
    count = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            eroded = 1
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if img[(i + di) * n + (j + dj)] == 0:
                        eroded = 0
                        break
                if eroded == 0:
                    break
            count += eroded

    return count
