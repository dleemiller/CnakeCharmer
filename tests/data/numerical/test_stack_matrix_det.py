"""Test stack_matrix_det equivalence."""

import pytest

from cnake_data.cy.numerical.stack_matrix_det import (
    stack_matrix_det as cy_func,
)
from cnake_data.py.numerical.stack_matrix_det import (
    stack_matrix_det as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_stack_matrix_det_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-6, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
