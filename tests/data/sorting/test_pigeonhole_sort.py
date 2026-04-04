"""Test pigeonhole_sort equivalence."""

import pytest

from cnake_data.cy.sorting.pigeonhole_sort import pigeonhole_sort as cy_func
from cnake_data.py.sorting.pigeonhole_sort import pigeonhole_sort as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_pigeonhole_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
