"""Test const dot product equivalence."""

import pytest

from cnake_data.cy.numerical.const_dot_product import (
    const_dot_product as cy_const_dot_product,
)
from cnake_data.py.numerical.const_dot_product import (
    const_dot_product as py_const_dot_product,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_const_dot_product_equivalence(n):
    py_result = py_const_dot_product(n)
    cy_result = cy_const_dot_product(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
