"""Test batch_factorial_mod equivalence."""

import pytest

from cnake_charmer.cy.math_problems.batch_factorial_mod import batch_factorial_mod as cy_func
from cnake_charmer.py.math_problems.batch_factorial_mod import batch_factorial_mod as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_batch_factorial_mod_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
