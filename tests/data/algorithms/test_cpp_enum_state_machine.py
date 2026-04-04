"""Test cpp_enum_state_machine equivalence between Python and Cython."""

import pytest

from cnake_data.cy.algorithms.cpp_enum_state_machine import cpp_enum_state_machine as cy_func
from cnake_data.py.algorithms.cpp_enum_state_machine import cpp_enum_state_machine as py_func


@pytest.mark.parametrize("n", [1000, 100000, 1000000])
def test_cpp_enum_state_machine_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
