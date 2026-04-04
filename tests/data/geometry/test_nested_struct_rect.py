"""Test nested_struct_rect equivalence."""

import pytest

from cnake_data.cy.geometry.nested_struct_rect import (
    nested_struct_rect as cy_func,
)
from cnake_data.py.geometry.nested_struct_rect import (
    nested_struct_rect as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 50000])
def test_nested_struct_rect_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
