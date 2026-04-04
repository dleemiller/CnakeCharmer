"""Test except_check_sqrt equivalence."""

import pytest

from cnake_data.cy.numerical.except_check_sqrt import except_check_sqrt as cy_except_check_sqrt
from cnake_data.py.numerical.except_check_sqrt import except_check_sqrt as py_except_check_sqrt


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_except_check_sqrt_equivalence(n):
    py_result = py_except_check_sqrt(n)
    cy_result = cy_except_check_sqrt(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
