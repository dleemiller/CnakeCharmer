"""Test gift_wrapping equivalence."""

import pytest

from cnake_data.cy.geometry.gift_wrapping import gift_wrapping as cy_func
from cnake_data.py.geometry.gift_wrapping import gift_wrapping as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_gift_wrapping_equivalence(n):
    assert py_func(n) == cy_func(n)
