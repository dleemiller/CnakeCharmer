"""Test polyline_simplify equivalence."""

import pytest

from cnake_data.cy.geometry.polyline_simplify import polyline_simplify as cy_func
from cnake_data.py.geometry.polyline_simplify import polyline_simplify as py_func


@pytest.mark.parametrize("n", [100, 500, 1000])
def test_polyline_simplify_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result[0] == cy_result[0]  # count must match exactly
    for p, c in zip(py_result[1:], cy_result[1:], strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
