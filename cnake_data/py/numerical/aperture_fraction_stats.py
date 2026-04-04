"""Estimate light fraction captured by a circular aperture.

Keywords: numerical, aperture photometry, gaussian psf, integration, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2.4, 3.2, 8, 5, 0.08))
def aperture_fraction_stats(
    fwhm: float,
    radius: float,
    max_pix_rad: int,
    piece: int,
    background: float,
) -> tuple:
    """Integrate a Gaussian PSF over sub-pixel bins and collect aperture metrics."""
    sigma2 = (fwhm / 2.35) ** 2
    radius2 = radius * radius
    bit = 1.0 / piece

    rad_sum = 0.0
    all_sum = 0.0
    max_pix_sum = 0.0

    for i in range(-max_pix_rad, max_pix_rad):
        for j in range(-max_pix_rad, max_pix_rad):
            pix_sum = 0.0
            for k in range(piece):
                x = (i - 0.5) + (k + 0.5) * bit
                fx = math.exp(-(x * x) / (2.0 * sigma2))
                for m in range(piece):
                    y = (j - 0.5) + (m + 0.5) * bit
                    fy = math.exp(-(y * y) / (2.0 * sigma2))
                    inten = fx * fy + background
                    this_bit = inten * bit * bit
                    pix_sum += this_bit
                    if x * x + y * y <= radius2:
                        rad_sum += this_bit
            all_sum += pix_sum
            if pix_sum > max_pix_sum:
                max_pix_sum = pix_sum

    ratio = rad_sum / all_sum
    max_pixel_fraction = max_pix_sum / all_sum
    weighted = ratio * 0.8 + max_pixel_fraction * 0.2
    return (ratio, max_pixel_fraction, weighted)
