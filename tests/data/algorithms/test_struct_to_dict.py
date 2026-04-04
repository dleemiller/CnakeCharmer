"""Test struct_to_dict equivalence."""

import pytest

from cnake_data.cy.algorithms.struct_to_dict import (
    struct_to_dict as cy_func,
)
from cnake_data.py.algorithms.struct_to_dict import (
    struct_to_dict as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 50000])
def test_struct_to_dict_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
