"""Test dot product equivalence."""

import pytest

from cnake_charmer.cy.numerical.dot_product import dot_product as cy_dot_product
from cnake_charmer.py.numerical.dot_product import dot_product as py_dot_product


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_dot_product_equivalence(n):
    py_result = py_dot_product(n)
    cy_result = cy_dot_product(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
