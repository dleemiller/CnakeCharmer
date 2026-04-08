# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bilinear mesh warp for grid-based image deformation — Cython implementation."""

from libc.math cimport cos, sin
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100, 100, 8))
def mesh_warp(int rows, int cols, int mesh_size):
    """Apply a mesh-based bilinear warp to a synthetic image."""
    cdef int m = mesh_size
    cdef double cell_h = (rows - 1.0) / (m - 1)
    cdef double cell_w = (cols - 1.0) / (m - 1)

    cdef double *mesh_sx = <double *>malloc(m * m * sizeof(double))
    cdef double *mesh_sy = <double *>malloc(m * m * sizeof(double))
    if not mesh_sx or not mesh_sy:
        free(mesh_sx); free(mesh_sy)
        raise MemoryError()

    cdef int i, j, r, c, i0, j0, sr, sc
    cdef double ci, cj, fy, fx, sx, sy, val

    for i in range(m):
        for j in range(m):
            mesh_sx[i * m + j] = j * cell_w + 4.0 * sin(i * 0.9 + j * 0.7)
            mesh_sy[i * m + j] = i * cell_h + 3.0 * cos(i * 0.6 - j * 0.8)

    cdef double total = 0.0, center_val = 0.0, max_sx = 0.0
    cdef int cr = rows // 2
    cdef int cc = cols // 2
    cdef double mfx, mfy

    for r in range(rows):
        ci = r / cell_h
        i0 = <int>ci
        if i0 >= m - 1:
            i0 = m - 2
        fy = ci - i0

        for c in range(cols):
            cj = c / cell_w
            j0 = <int>cj
            if j0 >= m - 1:
                j0 = m - 2
            fx = cj - j0
            mfx = 1.0 - fx
            mfy = 1.0 - fy

            sx = (mesh_sx[i0 * m + j0]       * mfx * mfy
                + mesh_sx[i0 * m + j0 + 1]   * fx  * mfy
                + mesh_sx[(i0+1) * m + j0]   * mfx * fy
                + mesh_sx[(i0+1) * m + j0+1] * fx  * fy)
            sy = (mesh_sy[i0 * m + j0]       * mfx * mfy
                + mesh_sy[i0 * m + j0 + 1]   * fx  * mfy
                + mesh_sy[(i0+1) * m + j0]   * mfx * fy
                + mesh_sy[(i0+1) * m + j0+1] * fx  * fy)

            sr = <int>(sy + 0.5)
            sc = <int>(sx + 0.5)
            if sr < 0:
                sr = 0
            elif sr >= rows:
                sr = rows - 1
            if sc < 0:
                sc = 0
            elif sc >= cols:
                sc = cols - 1

            val = sin(sr * 0.1) * cos(sc * 0.1)
            total += val
            if r == cr and c == cc:
                center_val = val
            if sx > max_sx:
                max_sx = sx

    free(mesh_sx); free(mesh_sy)
    return (total, center_val, max_sx)
