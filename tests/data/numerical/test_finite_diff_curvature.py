"""Test finite_diff_curvature equivalence."""

import pytest

from cnake_data.cy.numerical.finite_diff_curvature import finite_diff_curvature as cy_func
from cnake_data.py.numerical.finite_diff_curvature import finite_diff_curvature as py_func


@pytest.mark.parametrize("n", [3, 32, 256, 2048])
def test_finite_diff_curvature_equivalence(n):
    assert py_func(n) == cy_func(n)
