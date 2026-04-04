"""Test numerical_derivative equivalence."""

import pytest

from cnake_data.cy.numerical.numerical_derivative import numerical_derivative as cy_func
from cnake_data.py.numerical.numerical_derivative import numerical_derivative as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_numerical_derivative_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
