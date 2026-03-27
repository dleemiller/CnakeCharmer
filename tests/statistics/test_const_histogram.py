"""Test const histogram equivalence."""

import pytest

from cnake_charmer.cy.statistics.const_histogram import (
    const_histogram as cy_const_histogram,
)
from cnake_charmer.py.statistics.const_histogram import (
    const_histogram as py_const_histogram,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_const_histogram_equivalence(n):
    py_result = py_const_histogram(n)
    cy_result = cy_const_histogram(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
