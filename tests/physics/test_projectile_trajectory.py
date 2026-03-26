"""Test projectile_trajectory equivalence."""

import pytest

from cnake_charmer.cy.physics.projectile_trajectory import projectile_trajectory as cy_func
from cnake_charmer.py.physics.projectile_trajectory import projectile_trajectory as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_projectile_trajectory_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-6 + 1e-6, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
