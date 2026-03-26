"""2D histogram of deterministic point pairs.

Keywords: statistics, histogram, 2d, binning, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def histogram_2d(n: int) -> int:
    """Compute 2D histogram and return the maximum bin count.

    Points: x[i] = (i*7+3) % 100, y[i] = (i*13+7) % 100.
    Bins: 10x10 grid (each bin covers range 10).

    Args:
        n: Number of points.

    Returns:
        Maximum bin count across all bins.
    """
    bins = [[0] * 10 for _ in range(10)]

    for i in range(n):
        xval = (i * 7 + 3) % 100
        yval = (i * 13 + 7) % 100
        bx = xval // 10
        by = yval // 10
        bins[bx][by] += 1

    max_count = 0
    for bx in range(10):
        for by in range(10):
            if bins[bx][by] > max_count:
                max_count = bins[bx][by]

    return max_count
