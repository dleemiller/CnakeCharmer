"""Test line_segment_intersections equivalence."""

import pytest

from cnake_data.cy.geometry.line_segment_intersections import (
    line_segment_intersections as cy_func,
)
from cnake_data.py.geometry.line_segment_intersections import (
    line_segment_intersections as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_line_segment_intersections_equivalence(n):
    assert py_func(n) == cy_func(n)
