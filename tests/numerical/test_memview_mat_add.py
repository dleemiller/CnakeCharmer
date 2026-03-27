"""Test element-wise matrix addition equivalence."""

import pytest

from cnake_charmer.cy.numerical.memview_mat_add import (
    memview_mat_add as cy_memview_mat_add,
)
from cnake_charmer.py.numerical.memview_mat_add import (
    memview_mat_add as py_memview_mat_add,
)


@pytest.mark.parametrize("n", [10, 50, 100, 300])
def test_memview_mat_add_equivalence(n):
    py_result = py_memview_mat_add(n)
    cy_result = cy_memview_mat_add(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
