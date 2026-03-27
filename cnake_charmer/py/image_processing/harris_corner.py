"""Harris corner detector response on a deterministic image.

Keywords: image processing, harris, corner detection, gradient, response, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def harris_corner(n: int) -> tuple:
    """Compute Harris corner response on an n x n deterministic image.

    Generates a deterministic image with structure (checkerboard-like pattern),
    computes gradients using Sobel operators, then computes Harris response
    R = det(M) - k * trace(M)^2 for each pixel with Gaussian-weighted window.

    Args:
        n: Image dimension (n x n).

    Returns:
        Tuple of (num_corners, max_response, response_sum) where num_corners
        is the count of pixels with R > threshold.
    """
    # Generate deterministic image with corners
    image = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            # Checkerboard with smooth transitions creates corners
            val = ((i // 16) + (j // 16)) % 2
            noise = ((i * 7 + j * 13 + 3) % 17) / 17.0
            image[i * n + j] = val * 200.0 + noise * 50.0

    # Compute Sobel gradients Ix, Iy (skip 1-pixel border)
    Ix = [0.0] * (n * n)
    Iy = [0.0] * (n * n)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            # Sobel X
            gx = (
                -1.0 * image[(i - 1) * n + (j - 1)]
                + 1.0 * image[(i - 1) * n + (j + 1)]
                - 2.0 * image[i * n + (j - 1)]
                + 2.0 * image[i * n + (j + 1)]
                - 1.0 * image[(i + 1) * n + (j - 1)]
                + 1.0 * image[(i + 1) * n + (j + 1)]
            )
            # Sobel Y
            gy = (
                -1.0 * image[(i - 1) * n + (j - 1)]
                - 2.0 * image[(i - 1) * n + j]
                - 1.0 * image[(i - 1) * n + (j + 1)]
                + 1.0 * image[(i + 1) * n + (j - 1)]
                + 2.0 * image[(i + 1) * n + j]
                + 1.0 * image[(i + 1) * n + (j + 1)]
            )
            Ix[i * n + j] = gx
            Iy[i * n + j] = gy

    # Harris response with 3x3 window summation
    k = 0.04
    threshold = 1e6
    num_corners = 0
    max_response = -1e30
    response_sum = 0.0

    for i in range(2, n - 2):
        for j in range(2, n - 2):
            # Sum products over 3x3 window
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            for wi in range(-1, 2):
                for wj in range(-1, 2):
                    ix = Ix[(i + wi) * n + (j + wj)]
                    iy = Iy[(i + wi) * n + (j + wj)]
                    sxx += ix * ix
                    syy += iy * iy
                    sxy += ix * iy

            det = sxx * syy - sxy * sxy
            trace = sxx + syy
            R = det - k * trace * trace

            response_sum += R
            if max_response < R:
                max_response = R
            if threshold < R:
                num_corners += 1

    return (num_corners, max_response, response_sum)
