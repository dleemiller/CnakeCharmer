"""Test matrix_chain_order equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.matrix_chain_order import matrix_chain_order as cy_func
from cnake_charmer.py.dynamic_programming.matrix_chain_order import matrix_chain_order as py_func


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_matrix_chain_order_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
