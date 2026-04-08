# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""UV-parameterized texture sampling with bilinear interpolation — Cython implementation."""

from libc.math cimport cos, pi, sin
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100, 100, 64))
def uv_texture_sample(int rows, int cols, int tex_size):
    """Sample a procedural texture via sinusoidally-distorted UV coordinates."""
    cdef int T = tex_size
    cdef double *tex = <double *>malloc(T * T * sizeof(double))
    if not tex:
        raise MemoryError()

    cdef int i, j
    for i in range(T):
        for j in range(T):
            tex[i * T + j] = sin(i * pi / T) * cos(j * 2.0 * pi / T)

    cdef double total = 0.0, center_val = 0.0, max_val = -1e18
    cdef int cr = rows // 2
    cdef int cc = cols // 2
    cdef double u_base, v_base, u, v, uf, vf, fu, fv, val
    cdef int r, c, u0, v0

    for r in range(rows):
        v_base = <double>r / (rows - 1) if rows > 1 else 0.0
        for c in range(cols):
            u_base = <double>c / (cols - 1) if cols > 1 else 0.0

            u = u_base + 0.08 * sin(v_base * pi * 2.0)
            v = v_base + 0.08 * cos(u_base * pi * 2.0)
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0

            uf = u * (T - 1)
            vf = v * (T - 1)
            u0 = <int>uf
            v0 = <int>vf
            if u0 >= T - 1:
                u0 = T - 2
            if v0 >= T - 1:
                v0 = T - 2
            fu = uf - u0
            fv = vf - v0

            val = (
                tex[v0 * T + u0]           * (1.0 - fu) * (1.0 - fv)
                + tex[v0 * T + u0 + 1]     * fu          * (1.0 - fv)
                + tex[(v0 + 1) * T + u0]   * (1.0 - fu) * fv
                + tex[(v0 + 1) * T + u0 + 1] * fu        * fv
            )

            total += val
            if r == cr and c == cc:
                center_val = val
            if val > max_val:
                max_val = val

    free(tex)
    return (total, center_val, max_val)
