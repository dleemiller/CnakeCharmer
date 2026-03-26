"""Test gradient_descent equivalence."""

import pytest

from cnake_charmer.cy.optimization.gradient_descent import gradient_descent as cy_func
from cnake_charmer.py.optimization.gradient_descent import gradient_descent as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_gradient_descent_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
