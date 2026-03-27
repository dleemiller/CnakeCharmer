"""Test ring_buffer_mean equivalence."""

import pytest

from cnake_charmer.cy.algorithms.ring_buffer_mean import ring_buffer_mean as cy_func
from cnake_charmer.py.algorithms.ring_buffer_mean import ring_buffer_mean as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_ring_buffer_mean_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
