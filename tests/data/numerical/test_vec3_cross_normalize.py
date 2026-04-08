"""Test vec3_cross_normalize equivalence."""

import pytest

from cnake_data.cy.numerical.vec3_cross_normalize import vec3_cross_normalize as cy_func
from cnake_data.py.numerical.vec3_cross_normalize import vec3_cross_normalize as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 8000])
def test_vec3_cross_normalize_equivalence(n):
    py_total, py_first_norm, py_last_cx = py_func(n)
    cy_total, cy_first_norm, cy_last_cx = cy_func(n)
    assert py_total == cy_total
    assert py_first_norm == cy_first_norm
    assert py_last_cx == cy_last_cx
