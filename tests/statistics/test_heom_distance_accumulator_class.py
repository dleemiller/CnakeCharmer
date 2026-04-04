"""Test heom_distance_accumulator_class equivalence."""

import pytest

from cnake_charmer.cy.statistics.heom_distance_accumulator_class import (
    heom_distance_accumulator_class as cy_func,
)
from cnake_charmer.py.statistics.heom_distance_accumulator_class import (
    heom_distance_accumulator_class as py_func,
)


@pytest.mark.parametrize(
    "n_rows,n_cols,cat_stride,seed", [(50, 8, 2, 5), (90, 12, 3, 11), (120, 10, 5, 23)]
)
def test_heom_distance_accumulator_class_equivalence(n_rows, n_cols, cat_stride, seed):
    assert py_func(n_rows, n_cols, cat_stride, seed) == cy_func(n_rows, n_cols, cat_stride, seed)
