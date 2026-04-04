"""Test prange_dot_product equivalence."""

import pytest

from cnake_data.cy.numerical.prange_dot_product import (
    prange_dot_product as cy_func,
)
from cnake_data.py.numerical.prange_dot_product import (
    prange_dot_product as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_prange_dot_product_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
