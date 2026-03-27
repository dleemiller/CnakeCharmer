"""Test fused_minmax equivalence."""

import pytest

from cnake_charmer.cy.numerical.fused_minmax import fused_minmax as cy_fused_minmax
from cnake_charmer.py.numerical.fused_minmax import fused_minmax as py_fused_minmax


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_fused_minmax_equivalence(n):
    py_result = py_fused_minmax(n)
    cy_result = cy_fused_minmax(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
