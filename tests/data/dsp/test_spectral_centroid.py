"""Test spectral_centroid equivalence."""

import pytest

from cnake_data.cy.dsp.spectral_centroid import spectral_centroid as cy_spectral_centroid
from cnake_data.py.dsp.spectral_centroid import spectral_centroid as py_spectral_centroid


@pytest.mark.parametrize("n", [50, 100, 200, 500])
def test_spectral_centroid_equivalence(n):
    py_result = py_spectral_centroid(n)
    cy_result = cy_spectral_centroid(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
