"""Bilinear interpolation upsampling of a grayscale image by 2x.

Returns discriminating tuple of output pixel metrics.

Keywords: image processing, bilinear, interpolation, upsampling, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def bilinear_interpolation(n: int) -> tuple:
    """Upsample n x n image to (2n-1) x (2n-1) using bilinear interpolation.

    Source pixel[i][j] = (i*7 + j*13 + 3) % 256.

    Args:
        n: Source image dimension (n x n).

    Returns:
        Tuple of (corner_sum, center_val, total_sum).
    """
    # Generate source image
    src = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            src[i * n + j] = (i * 7 + j * 13 + 3) % 256

    out_n = 2 * n - 1
    out = [0] * (out_n * out_n)

    for oi in range(out_n):
        for oj in range(out_n):
            # Map output coords to source coords
            si = oi / 2.0
            sj = oj / 2.0

            i0 = int(si)
            j0 = int(sj)
            i1 = min(i0 + 1, n - 1)
            j1 = min(j0 + 1, n - 1)

            fi = si - i0
            fj = sj - j0

            val = (
                src[i0 * n + j0] * (1.0 - fi) * (1.0 - fj)
                + src[i1 * n + j0] * fi * (1.0 - fj)
                + src[i0 * n + j1] * (1.0 - fi) * fj
                + src[i1 * n + j1] * fi * fj
            )

            out[oi * out_n + oj] = int(val + 0.5)

    # corner_sum = sum of 4 corners
    corner_sum = (
        out[0] + out[out_n - 1] + out[(out_n - 1) * out_n] + out[(out_n - 1) * out_n + out_n - 1]
    )

    # center_val
    center = out[(out_n // 2) * out_n + out_n // 2]

    # total_sum
    total_sum = 0
    for i in range(out_n * out_n):
        total_sum += out[i]

    return (corner_sum, center, total_sum)
