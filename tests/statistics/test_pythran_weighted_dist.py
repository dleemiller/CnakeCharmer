"""Test pythran_weighted_dist equivalence."""

import pytest

from cnake_charmer.cy.statistics.pythran_weighted_dist import pythran_weighted_dist as cy_func
from cnake_charmer.py.statistics.pythran_weighted_dist import pythran_weighted_dist as py_func


@pytest.mark.parametrize("n", [1000, 5000, 50000])
def test_pythran_weighted_dist_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-6, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol
