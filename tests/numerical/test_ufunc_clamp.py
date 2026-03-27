"""Test ufunc_clamp equivalence."""

import pytest

from cnake_charmer.cy.numerical.ufunc_clamp import ufunc_clamp as cy_func
from cnake_charmer.py.numerical.ufunc_clamp import ufunc_clamp as py_func


@pytest.mark.parametrize("n", [100, 1000, 100000])
def test_ufunc_clamp_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) / max(abs(py_result), 1) < 1e-4, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
