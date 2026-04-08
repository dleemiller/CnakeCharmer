"""Bicubic interpolation for image upsampling.

Samples a deterministic source image at sub-pixel positions using the
cubic convolution kernel (Keys cubic), accumulating the interpolated values.

Keywords: bicubic interpolation, cubic convolution, image resampling, upsampling
"""

import math

from cnake_data.benchmarks import python_benchmark


def _cubic_weight(t: float) -> float:
    """Keys cubic convolution kernel weight for distance t."""
    t = abs(t)
    if t < 1.0:
        return 1.5 * t * t * t - 2.5 * t * t + 1.0
    if t < 2.0:
        return -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    return 0.0


@python_benchmark(args=(80, 80, 2))
def bicubic_interp(src_rows: int, src_cols: int, scale: int) -> tuple:
    """Bicubic upsample a deterministic image by integer scale factor.

    Source image: src[r][c] = sin(r * 0.3) * cos(c * 0.2)

    Args:
        src_rows: Source image height.
        src_cols: Source image width.
        scale: Integer upsampling factor.

    Returns:
        Tuple of (total_sum, center_val, corner_val).
    """
    dst_rows = src_rows * scale
    dst_cols = src_cols * scale

    total = 0.0
    center_val = 0.0
    corner_val = 0.0
    cr = dst_rows // 2
    cc = dst_cols // 2

    for dr in range(dst_rows):
        sy = dr / scale
        iy = int(sy)
        fy = sy - iy
        for dc in range(dst_cols):
            sx = dc / scale
            ix = int(sx)
            fx = sx - ix

            val = 0.0
            for m in range(-1, 3):
                wy = _cubic_weight(m - fy)
                sr = iy + m
                if sr < 0:
                    sr = 0
                elif sr >= src_rows:
                    sr = src_rows - 1
                for n in range(-1, 3):
                    wx = _cubic_weight(n - fx)
                    sc = ix + n
                    if sc < 0:
                        sc = 0
                    elif sc >= src_cols:
                        sc = src_cols - 1
                    src_val = math.sin(sr * 0.3) * math.cos(sc * 0.2)
                    val += wy * wx * src_val

            total += val
            if dr == cr and dc == cc:
                center_val = val
            if dr == 0 and dc == 0:
                corner_val = val

    return (total, center_val, corner_val)
