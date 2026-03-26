"""Test prefix max equivalence."""

import pytest

from cnake_charmer.cy.numerical.prefix_max import prefix_max as cy_prefix_max
from cnake_charmer.py.numerical.prefix_max import prefix_max as py_prefix_max


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_prefix_max_equivalence(n):
    assert py_prefix_max(n) == cy_prefix_max(n)
