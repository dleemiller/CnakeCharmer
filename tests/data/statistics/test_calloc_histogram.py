"""Test calloc_histogram equivalence."""

import pytest

from cnake_data.cy.statistics.calloc_histogram import (
    calloc_histogram as cy_func,
)
from cnake_data.py.statistics.calloc_histogram import (
    calloc_histogram as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_calloc_histogram_equivalence(n):
    assert py_func(n) == cy_func(n)
