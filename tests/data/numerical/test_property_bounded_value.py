"""Test property_bounded_value equivalence."""

import pytest

from cnake_data.cy.numerical.property_bounded_value import (
    property_bounded_value as cy_func,
)
from cnake_data.py.numerical.property_bounded_value import (
    property_bounded_value as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_property_bounded_value_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-6, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
