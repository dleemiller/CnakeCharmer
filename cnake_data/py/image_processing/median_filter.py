"""Apply 3x3 median filter to grayscale image.

Keywords: image processing, median filter, grayscale, smoothing, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def median_filter(n: int) -> int:
    """Apply 3x3 median filter to n x n grayscale image.

    Pixel[i][j] = (i*17 + j*31 + 5) % 256.
    For border pixels, the original value is kept.

    Args:
        n: Image dimension (n x n).

    Returns:
        Sum of all filtered pixel values.
    """
    size = n * n
    img = [0] * size
    out = [0] * size

    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 17 + j * 31 + 5) % 256

    # Copy border pixels
    for i in range(n):
        out[i] = img[i]  # top row
        out[(n - 1) * n + i] = img[(n - 1) * n + i]  # bottom row
    for i in range(n):
        out[i * n] = img[i * n]  # left col
        out[i * n + n - 1] = img[i * n + n - 1]  # right col

    # Apply median filter to interior
    window = [0] * 9
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            k = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    window[k] = img[(i + di) * n + (j + dj)]
                    k += 1
            # Insertion sort on 9 elements
            for a in range(1, 9):
                key = window[a]
                b = a - 1
                while b >= 0 and window[b] > key:
                    window[b + 1] = window[b]
                    b -= 1
                window[b + 1] = key
            out[i * n + j] = window[4]

    total = 0
    for i in range(size):
        total += out[i]
    return total
