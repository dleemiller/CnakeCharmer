"""Test range_iterator_sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.range_iterator_sum import range_iterator_sum as cy_func
from cnake_charmer.py.numerical.range_iterator_sum import range_iterator_sum as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_range_iterator_sum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
