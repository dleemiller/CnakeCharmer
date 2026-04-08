# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Flat-field (gain and offset) radiometric correction — Cython implementation."""

from libc.math cimport cos, sin

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200, 200))
def flat_field_correction(int rows, int cols):
    """Apply flat-field correction to a deterministically generated raw image."""
    cdef double mean_flat = 200.0
    cdef double total = 0.0, min_val = 1e18, max_val = -1e18
    cdef double raw, dark, flat, denom, corrected
    cdef int r, c

    for r in range(rows):
        for c in range(cols):
            raw = 128.0 + 64.0 * sin(r * 0.05) * cos(c * 0.05)
            dark = 10.0 + 2.0 * sin(r * 0.1 + c * 0.1)
            flat = 200.0 + 30.0 * cos(r * 0.07) * sin(c * 0.07)
            denom = flat - dark
            corrected = (raw - dark) / denom * mean_flat if denom != 0.0 else 0.0
            total += corrected
            if corrected < min_val:
                min_val = corrected
            if corrected > max_val:
                max_val = corrected

    return (total, min_val, max_val)
