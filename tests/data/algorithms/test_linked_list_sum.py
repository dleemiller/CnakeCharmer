"""Test linked_list_sum equivalence."""

import pytest

from cnake_data.cy.algorithms.linked_list_sum import linked_list_sum as cy_func
from cnake_data.py.algorithms.linked_list_sum import linked_list_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_linked_list_sum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
