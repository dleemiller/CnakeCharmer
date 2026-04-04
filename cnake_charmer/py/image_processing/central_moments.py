"""Raw image moments up to order 3 for a deterministic n×n image.

Keywords: image processing, moments, raw moments, statistics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def central_moments(n: int) -> tuple:
    """Compute raw image moments M[p][q] for p+q <= 3 on an n×n image.

    Image: pixel[r][c] = (r * 1009 + c * 2003 + 42) % 256
    Moments: M[p][q] = sum_r sum_c (r^p * c^q * pixel[r][c])

    Args:
        n: Image side length (n×n pixels).

    Returns:
        Tuple of (M[0][0], M[1][0], M[0][1], M[2][0]) as integers.
    """
    m00 = 0
    m10 = 0
    m01 = 0
    m20 = 0

    for r in range(n):
        for c in range(n):
            pixel = (r * 1009 + c * 2003 + 42) % 256
            m00 += pixel
            m10 += r * pixel
            m01 += c * pixel
            m20 += r * r * pixel

    return (m00, m10, m01, m20)
