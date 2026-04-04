"""Test distinct_subsequences equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.distinct_subsequences import (
    distinct_subsequences as cy_func,
)
from cnake_data.py.dynamic_programming.distinct_subsequences import (
    distinct_subsequences as py_func,
)


@pytest.mark.parametrize("n", [30, 90, 300, 600])
def test_distinct_subsequences_equivalence(n):
    assert py_func(n) == cy_func(n)
