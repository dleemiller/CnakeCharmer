"""Test diffusion_limited_agg equivalence."""

import pytest

from cnake_charmer.cy.simulation.diffusion_limited_agg import diffusion_limited_agg as cy_func
from cnake_charmer.py.simulation.diffusion_limited_agg import diffusion_limited_agg as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_diffusion_limited_agg_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
