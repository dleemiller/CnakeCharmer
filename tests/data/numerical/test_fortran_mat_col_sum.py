"""Test fortran_mat_col_sum equivalence."""

import pytest

from cnake_data.cy.numerical.fortran_mat_col_sum import (
    fortran_mat_col_sum as cy_func,
)
from cnake_data.py.numerical.fortran_mat_col_sum import (
    fortran_mat_col_sum as py_func,
)


@pytest.mark.parametrize("n", [5, 20, 50, 300])
def test_fortran_mat_col_sum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
