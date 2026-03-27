"""Test fused_dot_product equivalence."""

import pytest

from cnake_charmer.cy.numerical.fused_dot_product import fused_dot_product as cy_fused_dot_product
from cnake_charmer.py.numerical.fused_dot_product import fused_dot_product as py_fused_dot_product


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_fused_dot_product_equivalence(n):
    py_result = py_fused_dot_product(n)
    cy_result = cy_fused_dot_product(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
