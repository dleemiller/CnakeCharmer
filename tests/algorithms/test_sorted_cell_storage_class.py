"""Test sorted_cell_storage_class equivalence."""

import pytest

from cnake_charmer.cy.algorithms.sorted_cell_storage_class import (
    sorted_cell_storage_class as cy_func,
)
from cnake_charmer.py.algorithms.sorted_cell_storage_class import (
    sorted_cell_storage_class as py_func,
)


@pytest.mark.parametrize(
    "n_values,n_queries,seed,mod", [(90, 40, 7, 997), (300, 130, 17, 4093), (700, 220, 33, 10007)]
)
def test_sorted_cell_storage_class_equivalence(n_values, n_queries, seed, mod):
    assert py_func(n_values, n_queries, seed, mod) == cy_func(n_values, n_queries, seed, mod)
