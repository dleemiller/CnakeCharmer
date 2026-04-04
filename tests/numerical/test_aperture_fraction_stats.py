"""Test aperture_fraction_stats equivalence."""

import pytest

from cnake_charmer.cy.numerical.aperture_fraction_stats import aperture_fraction_stats as cy_func
from cnake_charmer.py.numerical.aperture_fraction_stats import aperture_fraction_stats as py_func


@pytest.mark.parametrize(
    "fwhm,radius,max_pix_rad,piece,background",
    [
        (2.1, 2.6, 5, 3, 0.02),
        (2.4, 3.2, 6, 4, 0.08),
        (1.8, 2.2, 7, 3, 0.01),
    ],
)
def test_aperture_fraction_stats_equivalence(fwhm, radius, max_pix_rad, piece, background):
    py_result = py_func(fwhm, radius, max_pix_rad, piece, background)
    cy_result = cy_func(fwhm, radius, max_pix_rad, piece, background)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-8
