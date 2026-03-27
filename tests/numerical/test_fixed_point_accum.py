"""Test fixed_point_accum equivalence."""

import pytest

from cnake_charmer.cy.numerical.fixed_point_accum import fixed_point_accum as cy_func
from cnake_charmer.py.numerical.fixed_point_accum import fixed_point_accum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_fixed_point_accum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
