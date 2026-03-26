"""Test max_product_subarray equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.max_product_subarray import (
    max_product_subarray as cy_func,
)
from cnake_charmer.py.dynamic_programming.max_product_subarray import (
    max_product_subarray as py_func,
)


@pytest.mark.parametrize("n", [50, 500, 5000, 10000])
def test_max_product_subarray_equivalence(n):
    assert py_func(n) == cy_func(n)
