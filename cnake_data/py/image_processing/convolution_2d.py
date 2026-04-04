"""2D convolution filter for image processing.

Keywords: convolution, filter, image processing, kernel, 2d
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80,))
def convolution_2d(n):
    """Apply a 3x3 convolution kernel to an n×n image.

    Args:
        n: Image dimension.

    Returns:
        Tuple of (total_sum, max_val, center_val).
    """
    kh = 3
    kw = 3

    # Generate deterministic image
    img = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(((i * 7 + j * 13 + 3) % 97) / 97.0)
        img.append(row)

    # Gaussian-like 3x3 kernel
    kernel = [
        [1.0 / 16, 2.0 / 16, 1.0 / 16],
        [2.0 / 16, 4.0 / 16, 2.0 / 16],
        [1.0 / 16, 2.0 / 16, 1.0 / 16],
    ]

    # Pad image
    pad = 1
    pn = n + 2 * pad
    padded = []
    for _i in range(pn):
        row = [0.0] * pn
        padded.append(row)
    for i in range(n):
        for j in range(n):
            padded[i + pad][j + pad] = img[i][j]

    # Convolve
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            tot = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    tot += padded[i + ki][j + kj] * kernel[ki][kj]
            row.append(tot)
        result.append(row)

    total_sum = 0.0
    max_val = 0.0
    for i in range(n):
        for j in range(n):
            total_sum += result[i][j]
            if result[i][j] > max_val:
                max_val = result[i][j]

    center = n // 2
    center_val = result[center][center]

    return (total_sum, max_val, center_val)
