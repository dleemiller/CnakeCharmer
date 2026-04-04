"""Test lucas_numbers equivalence."""

import pytest

from cnake_data.cy.math_problems.lucas_numbers import lucas_numbers as cy_func
from cnake_data.py.math_problems.lucas_numbers import lucas_numbers as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_lucas_numbers_equivalence(n):
    assert py_func(n) == cy_func(n)
