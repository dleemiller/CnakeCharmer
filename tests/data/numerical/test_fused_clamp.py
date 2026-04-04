"""Test fused_clamp equivalence."""

import pytest

from cnake_data.cy.numerical.fused_clamp import fused_clamp as cy_fused_clamp
from cnake_data.py.numerical.fused_clamp import fused_clamp as py_fused_clamp


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_fused_clamp_equivalence(n):
    py_result = py_fused_clamp(n)
    cy_result = cy_fused_clamp(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
