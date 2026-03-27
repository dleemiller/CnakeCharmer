"""Test line_segment_intersection equivalence."""

import pytest

from cnake_charmer.cy.geometry.line_segment_intersection import line_segment_intersection as cy_func
from cnake_charmer.py.geometry.line_segment_intersection import line_segment_intersection as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_line_segment_intersection_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result[0] == cy_result[0]
    for a, b in zip(py_result[1:], cy_result[1:], strict=False):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-4
