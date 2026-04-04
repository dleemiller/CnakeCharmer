"""Test circle_packing equivalence."""

import pytest

from cnake_data.cy.geometry.circle_packing import circle_packing as cy_func
from cnake_data.py.geometry.circle_packing import circle_packing as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_circle_packing_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert isinstance(py_result, tuple)
    assert isinstance(cy_result, tuple)
    # Check total area
    assert abs(py_result[0] - cy_result[0]) / max(abs(py_result[0]), 1e-12) < 1e-6, (
        f"Area mismatch: py={py_result[0]}, cy={cy_result[0]}"
    )
    # Check accepted count
    assert py_result[1] == cy_result[1], f"Count mismatch: py={py_result[1]}, cy={cy_result[1]}"
