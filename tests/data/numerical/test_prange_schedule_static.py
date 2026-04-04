"""Test prange_schedule_static equivalence."""

import pytest

from cnake_data.cy.numerical.prange_schedule_static import (
    prange_schedule_static as cy_func,
)
from cnake_data.py.numerical.prange_schedule_static import (
    prange_schedule_static as py_func,
)


@pytest.mark.parametrize("n", [1, 100, 1000, 10000])
def test_prange_schedule_static_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    rel = abs(py_result - cy_result) / max(abs(py_result), 1.0)
    assert rel < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
