"""Test derangement_count equivalence."""

import pytest

from cnake_data.cy.math_problems.derangement_count import derangement_count as cy_func
from cnake_data.py.math_problems.derangement_count import derangement_count as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_derangement_count_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
