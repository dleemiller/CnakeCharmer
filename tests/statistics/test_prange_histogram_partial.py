"""Test prange_histogram_partial equivalence."""

import pytest

from cnake_charmer.cy.statistics.prange_histogram_partial import (
    prange_histogram_partial as cy_func,
)
from cnake_charmer.py.statistics.prange_histogram_partial import (
    prange_histogram_partial as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_prange_histogram_partial_equivalence(n):
    assert py_func(n) == cy_func(n)
