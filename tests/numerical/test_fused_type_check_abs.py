"""Test fused_type_check_abs equivalence."""

import pytest

from cnake_charmer.cy.numerical.fused_type_check_abs import (
    fused_type_check_abs as cy_func,
)
from cnake_charmer.py.numerical.fused_type_check_abs import (
    fused_type_check_abs as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_fused_type_check_abs_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
