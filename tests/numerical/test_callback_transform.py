"""Test callback_transform equivalence."""

import pytest

from cnake_charmer.cy.numerical.callback_transform import (
    callback_transform as cy_callback_transform,
)
from cnake_charmer.py.numerical.callback_transform import (
    callback_transform as py_callback_transform,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_callback_transform_equivalence(n):
    py_result = py_callback_transform(n)
    cy_result = cy_callback_transform(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
