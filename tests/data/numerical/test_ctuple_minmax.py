"""Test ctuple_minmax equivalence."""

import pytest

from cnake_data.cy.numerical.ctuple_minmax import ctuple_minmax as cy_func
from cnake_data.py.numerical.ctuple_minmax import ctuple_minmax as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_ctuple_minmax_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
