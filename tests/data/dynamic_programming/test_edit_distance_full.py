"""Test edit_distance_full equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.edit_distance_full import edit_distance_full as cy_func
from cnake_data.py.dynamic_programming.edit_distance_full import edit_distance_full as py_func


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_edit_distance_full_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
