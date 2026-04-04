"""Test perlin_noise equivalence."""

import pytest

from cnake_data.cy.simulation.perlin_noise import perlin_noise as cy_func
from cnake_data.py.simulation.perlin_noise import perlin_noise as py_func


@pytest.mark.parametrize("n", [10, 30, 50])
def test_perlin_noise_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
