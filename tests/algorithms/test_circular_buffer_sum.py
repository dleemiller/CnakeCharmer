"""Test circular_buffer_sum equivalence."""

import pytest

from cnake_charmer.cy.algorithms.circular_buffer_sum import circular_buffer_sum as cy_func
from cnake_charmer.py.algorithms.circular_buffer_sum import circular_buffer_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_circular_buffer_sum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
