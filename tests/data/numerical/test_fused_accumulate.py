"""Test fused_accumulate equivalence."""

import pytest

from cnake_data.cy.numerical.fused_accumulate import fused_accumulate as cy_fused_accumulate
from cnake_data.py.numerical.fused_accumulate import fused_accumulate as py_fused_accumulate


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_fused_accumulate_equivalence(n):
    py_result = py_fused_accumulate(n)
    cy_result = cy_fused_accumulate(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
