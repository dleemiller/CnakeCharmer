"""Test ufunc_fused_abs_diff equivalence."""

import numpy as np
import pytest

from cnake_charmer.cy.numerical.ufunc_fused_abs_diff import (
    abs_diff_scalar,
)
from cnake_charmer.cy.numerical.ufunc_fused_abs_diff import (
    ufunc_fused_abs_diff as cy_func,
)
from cnake_charmer.py.numerical.ufunc_fused_abs_diff import (
    ufunc_fused_abs_diff as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 100000])
def test_ufunc_fused_abs_diff_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) / max(abs(py_result), 1) < 1e-4, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )


def test_ufunc_fused_abs_diff_integer():
    """Verify the ufunc also handles integer arrays correctly."""
    a = np.array([5, 2, 8, 1], dtype=np.int64)
    b = np.array([3, 7, 4, 9], dtype=np.int64)
    result = abs_diff_scalar(a, b)
    expected = np.abs(a - b)
    assert np.array_equal(result, expected), f"Integer mismatch: {result} vs {expected}"
