# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Estimate light fraction captured by a circular aperture (Cython)."""

from libc.math cimport exp

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2.4, 3.2, 8, 5, 0.08))
def aperture_fraction_stats(double fwhm, double radius, int max_pix_rad, int piece, double background):
    cdef double sigma2 = (fwhm / 2.35) * (fwhm / 2.35)
    cdef double radius2 = radius * radius
    cdef double bit = 1.0 / piece
    cdef double rad_sum = 0.0
    cdef double all_sum = 0.0
    cdef double max_pix_sum = 0.0
    cdef int i, j, k, l
    cdef double pix_sum, x, y, fx, fy, inten, this_bit

    for i in range(-max_pix_rad, max_pix_rad):
        for j in range(-max_pix_rad, max_pix_rad):
            pix_sum = 0.0
            for k in range(piece):
                x = (i - 0.5) + (k + 0.5) * bit
                fx = exp(-(x * x) / (2.0 * sigma2))
                for l in range(piece):
                    y = (j - 0.5) + (l + 0.5) * bit
                    fy = exp(-(y * y) / (2.0 * sigma2))
                    inten = fx * fy + background
                    this_bit = inten * bit * bit
                    pix_sum += this_bit
                    if x * x + y * y <= radius2:
                        rad_sum += this_bit
            all_sum += pix_sum
            if pix_sum > max_pix_sum:
                max_pix_sum = pix_sum

    return (rad_sum / all_sum, max_pix_sum / all_sum, (rad_sum / all_sum) * 0.8 + (max_pix_sum / all_sum) * 0.2)
