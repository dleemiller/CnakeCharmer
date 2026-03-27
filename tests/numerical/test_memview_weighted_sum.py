"""Test memview_weighted_sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.memview_weighted_sum import memview_weighted_sum as cy_func
from cnake_charmer.py.numerical.memview_weighted_sum import memview_weighted_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_memview_weighted_sum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
