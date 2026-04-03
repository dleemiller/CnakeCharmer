"""Test spectral_norm equivalence."""

import pytest

from cnake_charmer.cy.numerical.spectral_norm import spectral_norm as cy_func
from cnake_charmer.py.numerical.spectral_norm import spectral_norm as py_func


@pytest.mark.parametrize("n", [10, 50, 100])
def test_spectral_norm_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
