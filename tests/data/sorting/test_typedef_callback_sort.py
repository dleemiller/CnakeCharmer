"""Test typedef_callback_sort equivalence."""

import pytest

from cnake_data.cy.sorting.typedef_callback_sort import (
    typedef_callback_sort as cy_typedef_callback_sort,
)
from cnake_data.py.sorting.typedef_callback_sort import (
    typedef_callback_sort as py_typedef_callback_sort,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_typedef_callback_sort_equivalence(n):
    assert py_typedef_callback_sort(n) == cy_typedef_callback_sort(n)
