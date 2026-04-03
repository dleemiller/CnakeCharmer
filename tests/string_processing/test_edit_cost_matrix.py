"""Test edit_cost_matrix equivalence."""

import pytest

from cnake_charmer.cy.string_processing.edit_cost_matrix import edit_cost_matrix as cy_func
from cnake_charmer.py.string_processing.edit_cost_matrix import edit_cost_matrix as py_func


@pytest.mark.parametrize("n,sub_cost", [(10, 1), (50, 2), (100, 2)])
def test_edit_cost_matrix_equivalence(n, sub_cost):
    assert py_func(n, sub_cost) == cy_func(n, sub_cost)
