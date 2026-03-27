"""Test prange_vector_add equivalence."""

import pytest

from cnake_charmer.cy.numerical.prange_vector_add import (
    prange_vector_add as cy_func,
)
from cnake_charmer.py.numerical.prange_vector_add import (
    prange_vector_add as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_prange_vector_add_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
