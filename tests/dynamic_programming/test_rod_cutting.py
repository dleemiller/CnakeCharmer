"""Test rod_cutting equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.rod_cutting import rod_cutting as cy_func
from cnake_charmer.py.dynamic_programming.rod_cutting import rod_cutting as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_rod_cutting_equivalence(n):
    assert py_func(n) == cy_func(n)
