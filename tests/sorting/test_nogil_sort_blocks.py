"""Test nogil_sort_blocks equivalence."""

import pytest

from cnake_charmer.cy.sorting.nogil_sort_blocks import (
    nogil_sort_blocks as cy_func,
)
from cnake_charmer.py.sorting.nogil_sort_blocks import (
    nogil_sort_blocks as py_func,
)


@pytest.mark.parametrize("n", [1, 5, 10, 100])
def test_nogil_sort_blocks_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
