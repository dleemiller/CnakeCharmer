"""Test spring_damper equivalence."""

import pytest

from cnake_charmer.cy.physics.spring_damper import spring_damper as cy_func
from cnake_charmer.py.physics.spring_damper import spring_damper as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_spring_damper_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for i in range(3):
        assert abs(py_result[i] - cy_result[i]) / max(abs(py_result[i]), 1.0) < 1e-4, (
            f"Mismatch at index {i}: py={py_result[i]}, cy={cy_result[i]}"
        )
