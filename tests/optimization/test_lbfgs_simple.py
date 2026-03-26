"""Test lbfgs_simple equivalence."""

import pytest

from cnake_charmer.cy.optimization.lbfgs_simple import lbfgs_simple as cy_func
from cnake_charmer.py.optimization.lbfgs_simple import lbfgs_simple as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_lbfgs_simple_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
