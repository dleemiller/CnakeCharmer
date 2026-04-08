"""Bilinear mesh warp for grid-based image deformation.

Deforms an image by mapping each output pixel through a control-point mesh,
using bilinear interpolation within each mesh cell to find the source position.

Keywords: mesh warp, grid warp, bilinear, spatial transform, image deformation
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100, 100, 8))
def mesh_warp(rows: int, cols: int, mesh_size: int) -> tuple:
    """Apply a mesh-based bilinear warp to a synthetic image.

    Mesh grid has mesh_size×mesh_size control points with sinusoidal offsets.
    Source image: src[r][c] = sin(r * 0.1) * cos(c * 0.1)

    Args:
        rows: Output image height.
        cols: Output image width.
        mesh_size: Number of mesh control points per axis.

    Returns:
        Tuple of (total_sum, center_val, max_src_x).
    """
    m = mesh_size
    cell_h = (rows - 1) / (m - 1)
    cell_w = (cols - 1) / (m - 1)

    # Mesh control points with sinusoidal displacements
    # mesh_dx[i][j], mesh_dy[i][j] = offset at control point (i, j)
    mesh_sx = [[0.0] * m for _ in range(m)]
    mesh_sy = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            # Nominal position + small sinusoidal warp
            mesh_sx[i][j] = j * cell_w + 4.0 * math.sin(i * 0.9 + j * 0.7)
            mesh_sy[i][j] = i * cell_h + 3.0 * math.cos(i * 0.6 - j * 0.8)

    total = 0.0
    center_val = 0.0
    max_sx = 0.0
    cr = rows // 2
    cc = cols // 2

    for r in range(rows):
        # Which mesh row cell
        ci = r / cell_h
        i0 = int(ci)
        if i0 >= m - 1:
            i0 = m - 2
        fy = ci - i0

        for c in range(cols):
            cj = c / cell_w
            j0 = int(cj)
            if j0 >= m - 1:
                j0 = m - 2
            fx = cj - j0

            # Bilinear interpolation of source coordinates
            sx = (
                mesh_sx[i0][j0] * (1 - fx) * (1 - fy)
                + mesh_sx[i0][j0 + 1] * fx * (1 - fy)
                + mesh_sx[i0 + 1][j0] * (1 - fx) * fy
                + mesh_sx[i0 + 1][j0 + 1] * fx * fy
            )
            sy = (
                mesh_sy[i0][j0] * (1 - fx) * (1 - fy)
                + mesh_sy[i0][j0 + 1] * fx * (1 - fy)
                + mesh_sy[i0 + 1][j0] * (1 - fx) * fy
                + mesh_sy[i0 + 1][j0 + 1] * fx * fy
            )

            # Sample source image with nearest-neighbor
            sr = int(sy + 0.5)
            sc = int(sx + 0.5)
            if sr < 0:
                sr = 0
            elif sr >= rows:
                sr = rows - 1
            if sc < 0:
                sc = 0
            elif sc >= cols:
                sc = cols - 1

            val = math.sin(sr * 0.1) * math.cos(sc * 0.1)
            total += val
            if r == cr and c == cc:
                center_val = val
            if sx > max_sx:
                max_sx = sx

    return (total, center_val, max_sx)
