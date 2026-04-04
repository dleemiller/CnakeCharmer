"""Test union_int_float equivalence."""

import pytest

from cnake_data.cy.numerical.union_int_float import union_int_float as cy_func
from cnake_data.py.numerical.union_int_float import union_int_float as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_union_int_float_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-2, f"Mismatch: py={py_result}, cy={cy_result}"
