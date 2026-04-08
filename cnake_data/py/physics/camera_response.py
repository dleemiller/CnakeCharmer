"""Camera response function calibration.

Models a sensor's nonlinear response to linear irradiance via a cubic
polynomial CRF, then inverts it using Newton's method — the core step
in radiometric calibration for HDR imaging pipelines.

Keywords: camera response function, radiometric calibration, HDR, inverse CRF, photometric
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(150, 150))
def camera_response(rows: int, cols: int) -> tuple:
    """Apply and invert a polynomial camera response function.

    CRF: f(x) = x - 0.3*x² + 0.1*x³  (maps linear irradiance → pixel value).
    Inverse: Newton's method, 5 iterations, starting from f(x).

    Args:
        rows: Image height.
        cols: Image width.

    Returns:
        Tuple of (sum_linear, sum_nonlinear, n_clipped) where n_clipped counts
        pixels whose nonlinear value exceeds 0.9.
    """
    sum_linear = 0.0
    sum_nonlinear = 0.0
    n_clipped = 0

    for r in range(rows):
        for c in range(cols):
            # Synthetic linear irradiance in [0, 1]
            irr = (math.sin(r * 0.05) * 0.5 + 0.5) * (math.cos(c * 0.07) * 0.5 + 0.5)

            # Apply CRF: cam = irr - 0.3*irr^2 + 0.1*irr^3
            irr2 = irr * irr
            irr3 = irr2 * irr
            cam = irr - 0.3 * irr2 + 0.1 * irr3

            # Invert CRF via Newton: find x s.t. x - 0.3x^2 + 0.1x^3 = cam
            x = cam
            for _ in range(5):
                fx = x - 0.3 * x * x + 0.1 * x * x * x - cam
                dfx = 1.0 - 0.6 * x + 0.3 * x * x
                x = x - fx / dfx

            sum_linear += x
            sum_nonlinear += cam
            if cam > 0.9:
                n_clipped += 1

    return (sum_linear, sum_nonlinear, n_clipped)
