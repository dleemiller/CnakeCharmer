"""Test struct_return_minmax equivalence."""

import pytest

from cnake_data.cy.numerical.struct_return_minmax import (
    struct_return_minmax as cy_func,
)
from cnake_data.py.numerical.struct_return_minmax import (
    struct_return_minmax as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_struct_return_minmax_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
