"""Test ufunc_fused_sigmoid equivalence and float32 support."""

import numpy as np
import pytest

from cnake_charmer.cy.nn_ops.ufunc_fused_sigmoid import (
    fused_sigmoid_scalar,
)
from cnake_charmer.cy.nn_ops.ufunc_fused_sigmoid import (
    ufunc_fused_sigmoid as cy_func,
)
from cnake_charmer.py.nn_ops.ufunc_fused_sigmoid import ufunc_fused_sigmoid as py_func


@pytest.mark.parametrize("n", [100, 1000, 100000])
def test_ufunc_fused_sigmoid_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) / max(abs(py_result), 1) < 1e-4, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )


def test_ufunc_fused_sigmoid_float32():
    """Verify the ufunc also handles float32 arrays correctly."""
    rng = np.random.RandomState(42)
    arr32 = rng.standard_normal(10000).astype(np.float32)
    result32 = fused_sigmoid_scalar(arr32)
    assert result32.dtype == np.float32
    expected = 1.0 / (1.0 + np.exp(-arr32.astype(np.float64)))
    assert np.allclose(result32.astype(np.float64), expected, rtol=1e-4)
