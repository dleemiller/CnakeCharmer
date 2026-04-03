"""Test piecewise_interp equivalence."""

import pytest

from cnake_charmer.cy.numerical.piecewise_interp import (
    piecewise_interp as cy_piecewise_interp,
)
from cnake_charmer.py.numerical.piecewise_interp import (
    piecewise_interp as py_piecewise_interp,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_piecewise_interp_equivalence(n):
    py_result = py_piecewise_interp(n)
    cy_result = cy_piecewise_interp(n)
    assert len(py_result) == 2
    assert len(cy_result) == 2
    # Check sum
    rel_err_sum = abs(py_result[0] - cy_result[0]) / max(abs(py_result[0]), 1.0)
    assert rel_err_sum < 1e-4, f"Sum mismatch: py={py_result[0]}, cy={cy_result[0]}"
    # Check max
    rel_err_max = abs(py_result[1] - cy_result[1]) / max(abs(py_result[1]), 1.0)
    assert rel_err_max < 1e-4, f"Max mismatch: py={py_result[1]}, cy={cy_result[1]}"
