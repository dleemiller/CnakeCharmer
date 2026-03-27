"""Test struct_dict_roundtrip equivalence."""

import pytest

from cnake_charmer.cy.algorithms.struct_dict_roundtrip import (
    struct_dict_roundtrip as cy_func,
)
from cnake_charmer.py.algorithms.struct_dict_roundtrip import (
    struct_dict_roundtrip as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_struct_dict_roundtrip_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
