"""Test matrix_power equivalence."""

import pytest

from cnake_charmer.cy.math_problems.matrix_power import matrix_power as cy_func
from cnake_charmer.py.math_problems.matrix_power import matrix_power as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_matrix_power_equivalence(n):
    assert py_func(n) == cy_func(n)
