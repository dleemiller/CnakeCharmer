"""Grid coordinate transforms, bilinear sampling, and average-fill."""

from __future__ import annotations


def get_positions_vec(t, x, y):
    x0, y0, dx, dy, sx, sy = t
    n = len(x)
    i_out = [0.0] * n
    j_out = [0.0] * n
    for i in range(n):
        j_ = (dy * x[i] - dy * x0 + sx * y0 - sx * y[i]) / (dx * dy - sx * sy)
        i_ = (y[i] - y0 - j_ * sy) / dy
        j_out[i] = j_ - 0.5
        i_out[i] = i_ - 0.5
    return i_out, j_out


def sample_bilinear_double(i_arr, j_arr, z, na):
    m = len(z)
    n = len(z[0])
    out = [na] * len(i_arr)
    for k in range(len(i_arr)):
        i = i_arr[k]
        j = j_arr[k]
        i0 = int(i // 1.0) if i % 1 != 0 else int(i - 1.0 if i != 0 else i)
        i1 = int(i0 + 1)
        j0 = int(j // 1.0) if j % 1 != 0 else int(j - 1.0 if j != 0 else j)
        j1 = int(j0 + 1)
        if i0 >= 0 and i1 < m and j0 >= 0 and j1 < n:
            out[k] = (
                z[i0][j0] * (i1 - i) * (j1 - j)
                + z[i1][j0] * (i - i0) * (j1 - j)
                + z[i0][j1] * (i1 - i) * (j - j0)
                + z[i1][j1] * (i - i0) * (j - j0)
            )
    return out


def fillarray_double(array, i_idx, j_idx, z_vals, nodata_value):
    if len(i_idx) != len(j_idx) or len(i_idx) != len(z_vals):
        return 1
    ny = len(array)
    nx = len(array[0])
    counts = [[0 for _ in range(nx)] for _ in range(ny)]
    for idx in range(len(i_idx)):
        i = i_idx[idx]
        j = j_idx[idx]
        array[i][j] += z_vals[idx]
        counts[i][j] += 1
    for i in range(ny):
        for j in range(nx):
            array[i][j] = array[i][j] / counts[i][j] if counts[i][j] else nodata_value
    return 0
