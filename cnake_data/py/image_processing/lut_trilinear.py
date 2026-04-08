"""3D LUT (Lookup Table) with trilinear interpolation for color grading.

Applies a 3D color transform LUT to an image by trilinear interpolation
in RGB-cube space. Widely used in professional color grading pipelines.

Keywords: 3D LUT, trilinear interpolation, color grading, color transform, rendering
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80, 80, 17))
def lut_trilinear(rows: int, cols: int, lut_size: int) -> tuple:
    """Apply a 3D LUT with trilinear interpolation to a synthetic RGB image.

    The LUT encodes a warm tone shift: boost R, preserve G, roll off B.

    Args:
        rows: Image height.
        cols: Image width.
        lut_size: Number of LUT nodes per axis.

    Returns:
        Tuple of (sum_r, sum_g, sum_b) of output channel totals.
    """
    n = lut_size
    step = 1.0 / (n - 1)

    # Flatten LUT into three 1D arrays indexed by ri * n * n + gi * n + bi
    lut_r = [0.0] * (n * n * n)
    lut_g = [0.0] * (n * n * n)
    lut_b = [0.0] * (n * n * n)

    for ri in range(n):
        r_in = ri * step
        for gi in range(n):
            g_in = gi * step
            for bi in range(n):
                b_in = bi * step
                idx = ri * n * n + gi * n + bi
                lum = 0.299 * r_in + 0.587 * g_in + 0.114 * b_in
                lut_r[idx] = min(1.0, r_in * 1.1 + lum * 0.05)
                lut_g[idx] = min(1.0, g_in * 0.97 + lum * 0.03)
                lut_b[idx] = max(0.0, b_in * 0.88 - lum * 0.02)

    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0

    for row in range(rows):
        for col in range(cols):
            r_in = math.sin(row * 0.05) * 0.5 + 0.5
            g_in = math.cos(col * 0.05) * 0.5 + 0.5
            b_in = math.sin((row + col) * 0.03) * 0.5 + 0.5

            ri_f = r_in * (n - 1)
            gi_f = g_in * (n - 1)
            bi_f = b_in * (n - 1)
            r0 = int(ri_f)
            g0 = int(gi_f)
            b0 = int(bi_f)
            if r0 >= n - 1:
                r0 = n - 2
            if g0 >= n - 1:
                g0 = n - 2
            if b0 >= n - 1:
                b0 = n - 2
            r1 = r0 + 1
            g1 = g0 + 1
            b1 = b0 + 1

            fr = ri_f - r0
            fg = gi_f - g0
            fb = bi_f - b0
            mfr = 1.0 - fr
            mfg = 1.0 - fg
            mfb = 1.0 - fb

            w000 = mfr * mfg * mfb
            w100 = fr * mfg * mfb
            w010 = mfr * fg * mfb
            w110 = fr * fg * mfb
            w001 = mfr * mfg * fb
            w101 = fr * mfg * fb
            w011 = mfr * fg * fb
            w111 = fr * fg * fb

            sum_r += (
                w000 * lut_r[r0 * n * n + g0 * n + b0]
                + w100 * lut_r[r1 * n * n + g0 * n + b0]
                + w010 * lut_r[r0 * n * n + g1 * n + b0]
                + w110 * lut_r[r1 * n * n + g1 * n + b0]
                + w001 * lut_r[r0 * n * n + g0 * n + b1]
                + w101 * lut_r[r1 * n * n + g0 * n + b1]
                + w011 * lut_r[r0 * n * n + g1 * n + b1]
                + w111 * lut_r[r1 * n * n + g1 * n + b1]
            )
            sum_g += (
                w000 * lut_g[r0 * n * n + g0 * n + b0]
                + w100 * lut_g[r1 * n * n + g0 * n + b0]
                + w010 * lut_g[r0 * n * n + g1 * n + b0]
                + w110 * lut_g[r1 * n * n + g1 * n + b0]
                + w001 * lut_g[r0 * n * n + g0 * n + b1]
                + w101 * lut_g[r1 * n * n + g0 * n + b1]
                + w011 * lut_g[r0 * n * n + g1 * n + b1]
                + w111 * lut_g[r1 * n * n + g1 * n + b1]
            )
            sum_b += (
                w000 * lut_b[r0 * n * n + g0 * n + b0]
                + w100 * lut_b[r1 * n * n + g0 * n + b0]
                + w010 * lut_b[r0 * n * n + g1 * n + b0]
                + w110 * lut_b[r1 * n * n + g1 * n + b0]
                + w001 * lut_b[r0 * n * n + g0 * n + b1]
                + w101 * lut_b[r1 * n * n + g0 * n + b1]
                + w011 * lut_b[r0 * n * n + g1 * n + b1]
                + w111 * lut_b[r1 * n * n + g1 * n + b1]
            )

    return (sum_r, sum_g, sum_b)
