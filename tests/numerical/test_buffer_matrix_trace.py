"""Test buffer_matrix_trace equivalence."""

import pytest

from cnake_charmer.cy.numerical.buffer_matrix_trace import (
    buffer_matrix_trace as cy_buffer_matrix_trace,
)
from cnake_charmer.py.numerical.buffer_matrix_trace import (
    buffer_matrix_trace as py_buffer_matrix_trace,
)


@pytest.mark.parametrize("n", [5, 20, 50, 200])
def test_buffer_matrix_trace_equivalence(n):
    py_result = py_buffer_matrix_trace(n)
    cy_result = cy_buffer_matrix_trace(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
