"""Test diffusion_2d equivalence."""

import pytest

from cnake_charmer.cy.simulation.diffusion_2d import diffusion_2d as cy_func
from cnake_charmer.py.simulation.diffusion_2d import diffusion_2d as py_func


@pytest.mark.parametrize("n", [10, 20, 50])
def test_diffusion_2d_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
