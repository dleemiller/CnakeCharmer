"""Test nested_loop_sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.nested_loop_sum import nested_loop_sum as cy_nested
from cnake_charmer.py.numerical.nested_loop_sum import nested_loop_sum as py_nested


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_nested_loop_sum_equivalence(n):
    assert py_nested(n) == cy_nested(n)
