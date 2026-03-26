"""Test z_function equivalence."""

import pytest

from cnake_charmer.cy.string_processing.z_function import z_function as cy_func
from cnake_charmer.py.string_processing.z_function import z_function as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_z_function_equivalence(n):
    assert py_func(n) == cy_func(n)
