"""Test fused_memview_scale equivalence."""

import pytest

from cnake_charmer.cy.numerical.fused_memview_scale import (
    fused_memview_scale as cy_func,
)
from cnake_charmer.py.numerical.fused_memview_scale import (
    fused_memview_scale as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_fused_memview_scale_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # Float precision is lower, use larger tolerance
    assert abs(py_result - cy_result) < max(abs(py_result) * 1e-4, 1e-2), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
