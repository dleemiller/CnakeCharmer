"""Test bisection_batch equivalence."""

import pytest

from cnake_data.cy.optimization.bisection_batch import bisection_batch as cy_func
from cnake_data.py.optimization.bisection_batch import bisection_batch as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_bisection_batch_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
