"""Test dynamic_array_grow equivalence."""

import pytest

from cnake_charmer.cy.algorithms.dynamic_array_grow import (
    dynamic_array_grow as cy_func,
)
from cnake_charmer.py.algorithms.dynamic_array_grow import (
    dynamic_array_grow as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_dynamic_array_grow_equivalence(n):
    assert py_func(n) == cy_func(n)
