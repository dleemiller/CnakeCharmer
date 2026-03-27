"""Test struct_return_stats equivalence."""

import pytest

from cnake_charmer.cy.statistics.struct_return_stats import (
    struct_return_stats as cy_func,
)
from cnake_charmer.py.statistics.struct_return_stats import (
    struct_return_stats as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_struct_return_stats_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
