"""Test numpy_run_length equivalence."""

import pytest

from cnake_charmer.cy.string_processing.numpy_run_length import (
    numpy_run_length as cy_func,
)
from cnake_charmer.py.string_processing.numpy_run_length import (
    numpy_run_length as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_numpy_run_length_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
