"""Test fused_array_sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.fused_array_sum import fused_array_sum as cy_fused_array_sum
from cnake_charmer.py.numerical.fused_array_sum import fused_array_sum as py_fused_array_sum


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_fused_array_sum_equivalence(n):
    py_result = py_fused_array_sum(n)
    cy_result = cy_fused_array_sum(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
