"""Test memcmp_dedup_count equivalence."""

import pytest

from cnake_data.cy.algorithms.memcmp_dedup_count import (
    memcmp_dedup_count as cy_func,
)
from cnake_data.py.algorithms.memcmp_dedup_count import (
    memcmp_dedup_count as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_memcmp_dedup_count_equivalence(n):
    assert py_func(n) == cy_func(n)
