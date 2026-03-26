"""Test floyd_cycle equivalence."""

import pytest

from cnake_charmer.cy.algorithms.floyd_cycle import floyd_cycle as cy_func
from cnake_charmer.py.algorithms.floyd_cycle import floyd_cycle as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_floyd_cycle_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
