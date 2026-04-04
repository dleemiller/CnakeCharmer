"""Test modular_exponentiation equivalence."""

import pytest

from cnake_data.cy.math_problems.modular_exponentiation import modular_exponentiation as cy_func
from cnake_data.py.math_problems.modular_exponentiation import modular_exponentiation as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_modular_exponentiation_equivalence(n):
    assert py_func(n) == cy_func(n)
