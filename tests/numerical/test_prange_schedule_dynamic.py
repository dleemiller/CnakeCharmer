"""Test prange_schedule_dynamic equivalence."""

import pytest

from cnake_charmer.cy.numerical.prange_schedule_dynamic import (
    prange_schedule_dynamic as cy_func,
)
from cnake_charmer.py.numerical.prange_schedule_dynamic import (
    prange_schedule_dynamic as py_func,
)


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_prange_schedule_dynamic_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    rel = abs(py_result - cy_result) / max(abs(py_result), 1.0)
    assert rel < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
