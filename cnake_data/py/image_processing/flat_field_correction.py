"""Flat-field (gain and offset) radiometric correction.

Applies per-pixel gain and dark-frame subtraction to linearize sensor
response. Computes statistics on the corrected image.

Keywords: flat field, gain correction, dark frame, radiometric calibration, photometric
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200, 200))
def flat_field_correction(rows: int, cols: int) -> tuple:
    """Apply flat-field correction to a deterministically generated raw image.

    Raw image:  raw[r][c] = 128 + 64 * sin(r * 0.05) * cos(c * 0.05)
    Dark frame: dark[r][c] = 10 + 2 * sin(r * 0.1 + c * 0.1)
    Flat field: flat[r][c] = 200 + 30 * cos(r * 0.07) * sin(c * 0.07)

    Correction: corrected = (raw - dark) / (flat - dark) * mean_flat

    Args:
        rows: Image height.
        cols: Image width.

    Returns:
        Tuple of (total_corrected, min_val, max_val).
    """
    mean_flat = 200.0  # target mean (known from calibration)

    total = 0.0
    min_val = 1e18
    max_val = -1e18

    for r in range(rows):
        for c in range(cols):
            raw = 128.0 + 64.0 * math.sin(r * 0.05) * math.cos(c * 0.05)
            dark = 10.0 + 2.0 * math.sin(r * 0.1 + c * 0.1)
            flat = 200.0 + 30.0 * math.cos(r * 0.07) * math.sin(c * 0.07)
            denom = flat - dark
            corrected = (raw - dark) / denom * mean_flat if denom != 0.0 else 0.0
            total += corrected
            if corrected < min_val:
                min_val = corrected
            if corrected > max_val:
                max_val = corrected

    return (total, min_val, max_val)
