"""Test lotka_volterra_rk4 equivalence."""

import pytest

from cnake_data.cy.diff_equations.lotka_volterra_rk4 import lotka_volterra_rk4 as cy_func
from cnake_data.py.diff_equations.lotka_volterra_rk4 import lotka_volterra_rk4 as py_func


@pytest.mark.parametrize("n_steps", [100, 500, 1000, 2000])
def test_lotka_volterra_rk4_equivalence(n_steps):
    assert py_func(n_steps) == cy_func(n_steps)
