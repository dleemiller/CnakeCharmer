"""Test romberg_integration equivalence."""

import pytest

from cnake_charmer.cy.numerical.romberg_integration import romberg_integration as cy_func
from cnake_charmer.py.numerical.romberg_integration import romberg_integration as py_func


@pytest.mark.parametrize("n", [3, 5, 10, 15])
def test_romberg_integration_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
