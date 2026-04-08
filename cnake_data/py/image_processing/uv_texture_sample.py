"""UV-parameterized texture sampling with bilinear interpolation.

Maps each output pixel to a UV coordinate via a sinusoidal distortion,
then samples a precomputed procedural texture using bilinear interpolation.
Models the texture sampling stage in UV-unwrapped rendering pipelines.

Keywords: UV mapping, texture sampling, bilinear interpolation, UV unwrapping, rendering
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100, 100, 64))
def uv_texture_sample(rows: int, cols: int, tex_size: int) -> tuple:
    """Sample a procedural texture via sinusoidally-distorted UV coordinates.

    UV base: normalized pixel coords.  Distortion: sinusoidal warp.
    Texture: T[i][j] = sin(i*π/T) * cos(j*2π/T).

    Args:
        rows: Output image height.
        cols: Output image width.
        tex_size: Texture resolution (tex_size × tex_size).

    Returns:
        Tuple of (total_sum, center_val, max_val).
    """
    T = tex_size
    tex = [0.0] * (T * T)
    for i in range(T):
        for j in range(T):
            tex[i * T + j] = math.sin(i * math.pi / T) * math.cos(j * 2.0 * math.pi / T)

    total = 0.0
    center_val = 0.0
    max_val = -1e18
    cr = rows // 2
    cc = cols // 2

    for r in range(rows):
        v_base = r / (rows - 1) if rows > 1 else 0.0
        for c in range(cols):
            u_base = c / (cols - 1) if cols > 1 else 0.0

            # Sinusoidal UV distortion
            u = u_base + 0.08 * math.sin(v_base * math.pi * 2.0)
            v = v_base + 0.08 * math.cos(u_base * math.pi * 2.0)
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0

            # Map to texture coords
            uf = u * (T - 1)
            vf = v * (T - 1)
            u0 = int(uf)
            v0 = int(vf)
            if u0 >= T - 1:
                u0 = T - 2
            if v0 >= T - 1:
                v0 = T - 2
            fu = uf - u0
            fv = vf - v0

            # Bilinear sample
            val = (
                tex[v0 * T + u0] * (1.0 - fu) * (1.0 - fv)
                + tex[v0 * T + u0 + 1] * fu * (1.0 - fv)
                + tex[(v0 + 1) * T + u0] * (1.0 - fu) * fv
                + tex[(v0 + 1) * T + u0 + 1] * fu * fv
            )

            total += val
            if r == cr and c == cc:
                center_val = val
            if val > max_val:
                max_val = val

    return (total, center_val, max_val)
