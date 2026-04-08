"""Gaussian PSF aperture photometry via sub-pixel integration.

For a circular Gaussian point-spread function, computes the fraction of
total light falling within a given aperture radius by sub-dividing each
pixel into piece×piece sub-pixels and summing contributions.

Keywords: aperture photometry, gaussian psf, sub-pixel, image processing, light fraction
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2.5, 3.0, 8, 8))
def gaussian_psf_aperture(fwhm: float, radius: float, max_pix_rad: int, piece: int) -> tuple:
    """Integrate Gaussian PSF over sub-pixel bins; return aperture metrics.

    Args:
        fwhm: Full-width at half-maximum of the PSF (pixels).
        radius: Aperture radius (pixels).
        max_pix_rad: Half-size of the pixel grid to sum over.
        piece: Number of sub-pixel divisions per pixel side.

    Returns:
        Tuple of (ratio, max_pixel_fraction) where ratio is the fraction of
        light within the aperture and max_pixel_fraction is the brightest
        pixel's fraction of total light.
    """
    sigma2 = (fwhm / 2.35) ** 2
    radius2 = radius * radius
    bit = 1.0 / piece
    psf_center_x = 0.5
    psf_center_y = 0.5

    rad_sum = 0.0
    all_sum = 0.0
    max_pix_sum = 0.0

    for i in range(-max_pix_rad, max_pix_rad):
        for j in range(-max_pix_rad, max_pix_rad):
            pix_sum = 0.0
            for k in range(piece):
                x = (i - psf_center_x) + (k + 0.5) * bit
                fx = math.exp(-(x * x) / (2.0 * sigma2))
                for ll in range(piece):
                    y = (j - psf_center_y) + (ll + 0.5) * bit
                    fy = math.exp(-(y * y) / (2.0 * sigma2))
                    this_bit = fx * fy * bit * bit
                    pix_sum += this_bit
                    if x * x + y * y <= radius2:
                        rad_sum += this_bit
            all_sum += pix_sum
            if pix_sum > max_pix_sum:
                max_pix_sum = pix_sum

    ratio = rad_sum / all_sum
    max_pixel_fraction = max_pix_sum / all_sum
    return (ratio, max_pixel_fraction)
