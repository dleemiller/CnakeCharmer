# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gaussian PSF aperture photometry via sub-pixel integration — Cython implementation."""

from libc.math cimport exp

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2.5, 3.0, 8, 8))
def gaussian_psf_aperture(double fwhm, double radius, int max_pix_rad, int piece):
    """Integrate Gaussian PSF over sub-pixel bins; return aperture metrics.

    Args:
        fwhm: Full-width at half-maximum of the PSF (pixels).
        radius: Aperture radius (pixels).
        max_pix_rad: Half-size of the pixel grid to sum over.
        piece: Number of sub-pixel divisions per pixel side.

    Returns:
        Tuple of (ratio, max_pixel_fraction).
    """
    cdef double sigma2 = (fwhm / 2.35) * (fwhm / 2.35)
    cdef double radius2 = radius * radius
    cdef double bit = 1.0 / piece
    cdef double psf_center_x = 0.5
    cdef double psf_center_y = 0.5

    cdef double rad_sum = 0.0
    cdef double all_sum = 0.0
    cdef double max_pix_sum = 0.0
    cdef double pix_sum, x, fx, y, fy, this_bit
    cdef int i, j, k, ll

    for i in range(-max_pix_rad, max_pix_rad):
        for j in range(-max_pix_rad, max_pix_rad):
            pix_sum = 0.0
            for k in range(piece):
                x = (i - psf_center_x) + (k + 0.5) * bit
                fx = exp(-(x * x) / (2.0 * sigma2))
                for ll in range(piece):
                    y = (j - psf_center_y) + (ll + 0.5) * bit
                    fy = exp(-(y * y) / (2.0 * sigma2))
                    this_bit = fx * fy * bit * bit
                    pix_sum += this_bit
                    if x * x + y * y <= radius2:
                        rad_sum += this_bit
            all_sum += pix_sum
            if pix_sum > max_pix_sum:
                max_pix_sum = pix_sum

    cdef double ratio = rad_sum / all_sum
    cdef double max_pixel_fraction = max_pix_sum / all_sum
    return (ratio, max_pixel_fraction)
