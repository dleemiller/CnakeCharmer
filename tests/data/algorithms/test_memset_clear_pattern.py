"""Test memset_clear_pattern equivalence."""

import pytest

from cnake_data.cy.algorithms.memset_clear_pattern import (
    memset_clear_pattern as cy_func,
)
from cnake_data.py.algorithms.memset_clear_pattern import (
    memset_clear_pattern as py_func,
)


@pytest.mark.parametrize("n", [64, 1000, 10000, 50000])
def test_memset_clear_pattern_equivalence(n):
    assert py_func(n) == cy_func(n)
