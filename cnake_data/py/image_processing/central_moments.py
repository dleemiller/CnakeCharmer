"""Compute central moments of a 2D image up to order 3.

Keywords: image processing, central moments, statistics, moment computation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80,))
def central_moments(n: int) -> tuple:
    """Compute central image moments mu_pq up to order 3 on an n x n image.

    Generates a deterministic n x n grayscale image, computes the centroid,
    then computes central moments mu_pq = sum(val * (c-cc)^p * (r-cr)^q)
    for all p+q <= 3.

    Args:
        n: Image dimension (n x n).

    Returns:
        Tuple of (mu_20, mu_02, mu_11, mu_00).
    """
    # Generate deterministic image
    img = [[0.0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            img[r][c] = float((r * 7 + c * 13 + 42) % 256)

    # Compute centroid
    sum_val = 0.0
    sum_r = 0.0
    sum_c = 0.0
    for r in range(n):
        for c in range(n):
            val = img[r][c]
            sum_val += val
            sum_r += r * val
            sum_c += c * val

    cr = sum_r / sum_val
    cc = sum_c / sum_val

    # Compute central moments up to order 3 in a 4x4 array
    mu = [[0.0] * 4 for _ in range(4)]
    for r in range(n):
        for c in range(n):
            val = img[r][c]
            dr = r - cr
            dc = c - cc
            for p in range(4):
                for q in range(4):
                    if p + q <= 3:
                        mu[p][q] += val * (dc**p) * (dr**q)

    return (mu[2][0], mu[0][2], mu[1][1], mu[0][0])
