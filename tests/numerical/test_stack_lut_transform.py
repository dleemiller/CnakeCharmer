"""Test stack_lut_transform equivalence."""

import pytest

from cnake_charmer.cy.numerical.stack_lut_transform import (
    stack_lut_transform as cy_func,
)
from cnake_charmer.py.numerical.stack_lut_transform import (
    stack_lut_transform as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_stack_lut_transform_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-6, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
