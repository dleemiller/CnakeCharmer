"""Test memoryview copy transform equivalence."""

import pytest

from cnake_charmer.cy.numerical.memview_copy_transform import (
    memview_copy_transform as cy_memview_copy_transform,
)
from cnake_charmer.py.numerical.memview_copy_transform import (
    memview_copy_transform as py_memview_copy_transform,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_memview_copy_transform_equivalence(n):
    py_result = py_memview_copy_transform(n)
    cy_result = cy_memview_copy_transform(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
