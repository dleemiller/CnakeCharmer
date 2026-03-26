"""Otsu's threshold on a grayscale image.

Keywords: image processing, otsu, threshold, histogram, segmentation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def threshold(n: int) -> int:
    """Compute optimal threshold via Otsu's method on n x n grayscale image.

    Pixel[i][j] = (i*17 + j*31 + 5) % 256. Builds histogram and finds
    the threshold that maximizes between-class variance.

    Args:
        n: Image dimension (n x n).

    Returns:
        Optimal threshold value (0-255).
    """
    # Build histogram
    hist = [0] * 256
    for i in range(n):
        for j in range(n):
            pixel = (i * 17 + j * 31 + 5) % 256
            hist[pixel] += 1

    total = n * n
    sum_all = 0.0
    for i in range(256):
        sum_all += i * hist[i]

    best_thresh = 0
    best_var = 0.0
    sum_bg = 0.0
    weight_bg = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    return best_thresh
