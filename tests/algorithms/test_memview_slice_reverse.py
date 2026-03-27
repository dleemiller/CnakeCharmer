"""Test memoryview slice reverse equivalence."""

import pytest

from cnake_charmer.cy.algorithms.memview_slice_reverse import (
    memview_slice_reverse as cy_memview_slice_reverse,
)
from cnake_charmer.py.algorithms.memview_slice_reverse import (
    memview_slice_reverse as py_memview_slice_reverse,
)


@pytest.mark.parametrize("n", [128, 1000, 10000, 100000])
def test_memview_slice_reverse_equivalence(n):
    py_result = py_memview_slice_reverse(n)
    cy_result = cy_memview_slice_reverse(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
