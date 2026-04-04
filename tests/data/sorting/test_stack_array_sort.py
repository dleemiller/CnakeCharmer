"""Test stack_array_sort equivalence."""

import pytest

from cnake_data.cy.sorting.stack_array_sort import (
    stack_array_sort as cy_func,
)
from cnake_data.py.sorting.stack_array_sort import (
    stack_array_sort as py_func,
)


@pytest.mark.parametrize("n", [1024, 2048, 10240])
def test_stack_array_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
