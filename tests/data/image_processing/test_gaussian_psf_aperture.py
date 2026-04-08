"""Test gaussian_psf_aperture equivalence."""

import pytest

from cnake_data.cy.image_processing.gaussian_psf_aperture import gaussian_psf_aperture as cy_func
from cnake_data.py.image_processing.gaussian_psf_aperture import gaussian_psf_aperture as py_func


@pytest.mark.parametrize(
    "fwhm,radius,max_pix_rad,piece",
    [
        (2.0, 2.5, 6, 4),
        (2.5, 3.0, 8, 8),
        (3.0, 3.5, 7, 6),
        (1.5, 2.0, 5, 4),
    ],
)
def test_gaussian_psf_aperture_equivalence(fwhm, radius, max_pix_rad, piece):
    assert py_func(fwhm, radius, max_pix_rad, piece) == cy_func(fwhm, radius, max_pix_rad, piece)
