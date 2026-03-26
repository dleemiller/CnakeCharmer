"""Test spring_mass equivalence."""

import pytest

from cnake_charmer.cy.simulation.spring_mass import spring_mass as cy_func
from cnake_charmer.py.simulation.spring_mass import spring_mass as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_spring_mass_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
