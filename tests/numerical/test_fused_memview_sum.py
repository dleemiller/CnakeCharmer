"""Test fused_memview_sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.fused_memview_sum import (
    fused_memview_sum as cy_func,
)
from cnake_charmer.py.numerical.fused_memview_sum import (
    fused_memview_sum as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_fused_memview_sum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
