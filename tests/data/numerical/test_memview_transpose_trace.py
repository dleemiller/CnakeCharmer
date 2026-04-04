"""Test memview_transpose_trace equivalence."""

import pytest

from cnake_data.cy.numerical.memview_transpose_trace import (
    memview_transpose_trace as cy_func,
)
from cnake_data.py.numerical.memview_transpose_trace import (
    memview_transpose_trace as py_func,
)


@pytest.mark.parametrize("n", [5, 20, 50, 200])
def test_memview_transpose_trace_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
