"""Test matrix transpose trace equivalence."""

import pytest

from cnake_data.cy.numerical.memview_mat_transpose import (
    memview_mat_transpose as cy_memview_mat_transpose,
)
from cnake_data.py.numerical.memview_mat_transpose import (
    memview_mat_transpose as py_memview_mat_transpose,
)


@pytest.mark.parametrize("n", [10, 50, 100, 300])
def test_memview_mat_transpose_equivalence(n):
    py_result = py_memview_mat_transpose(n)
    cy_result = cy_memview_mat_transpose(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
