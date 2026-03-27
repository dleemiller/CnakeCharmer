"""Test numpy_typed_histogram equivalence."""

import pytest

from cnake_charmer.cy.statistics.numpy_typed_histogram import (
    numpy_typed_histogram as cy_func,
)
from cnake_charmer.py.statistics.numpy_typed_histogram import (
    numpy_typed_histogram as py_func,
)


@pytest.mark.parametrize("n", [1000, 10000, 50000])
def test_numpy_typed_histogram_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
