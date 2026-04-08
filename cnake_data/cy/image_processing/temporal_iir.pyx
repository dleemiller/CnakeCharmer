# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Per-pixel recursive temporal IIR filter — Cython implementation."""

from libc.math cimport cos, sin
from libc.stdlib cimport calloc, free

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(80, 80, 30, 0.3))
def temporal_iir(int rows, int cols, int n_frames, double alpha):
    """Apply per-pixel IIR temporal smoothing over n_frames synthetic frames."""
    cdef double beta = 1.0 - alpha
    cdef double *state = <double *>calloc(rows * cols, sizeof(double))
    if not state:
        raise MemoryError()

    cdef int f, r, c
    cdef double val, v
    cdef double total_final = 0.0, max_final = -1e18, min_final = 1e18

    for f in range(n_frames):
        for r in range(rows):
            for c in range(cols):
                val = sin(r * 0.1 + f * 0.2) * cos(c * 0.1 - f * 0.15)
                state[r * cols + c] = alpha * val + beta * state[r * cols + c]

    for r in range(rows):
        for c in range(cols):
            v = state[r * cols + c]
            total_final += v
            if v > max_final:
                max_final = v
            if v < min_final:
                min_final = v

    free(state)
    return (total_final, max_final, min_final)
