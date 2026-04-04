"""Test buffer_sum_squares equivalence."""

import pytest

from cnake_data.cy.numerical.buffer_sum_squares import (
    buffer_sum_squares as cy_buffer_sum_squares,
)
from cnake_data.py.numerical.buffer_sum_squares import (
    buffer_sum_squares as py_buffer_sum_squares,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_buffer_sum_squares_equivalence(n):
    py_result = py_buffer_sum_squares(n)
    cy_result = cy_buffer_sum_squares(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
