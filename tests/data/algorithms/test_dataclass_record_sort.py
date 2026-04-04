"""Test dataclass_record_sort equivalence."""

import pytest

from cnake_data.cy.algorithms.dataclass_record_sort import dataclass_record_sort as cy_func
from cnake_data.py.algorithms.dataclass_record_sort import dataclass_record_sort as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_dataclass_record_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
