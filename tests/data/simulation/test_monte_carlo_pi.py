"""Test monte_carlo_pi equivalence."""

import pytest

from cnake_data.cy.simulation.monte_carlo_pi import monte_carlo_pi as cy_func
from cnake_data.py.simulation.monte_carlo_pi import monte_carlo_pi as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_monte_carlo_pi_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # Float comparison for estimate and last_point_distance
    assert abs(py_result[0] - cy_result[0]) / max(abs(py_result[0]), 1.0) < 1e-4
    assert py_result[1] == cy_result[1]
    assert abs(py_result[2] - cy_result[2]) / max(abs(py_result[2]), 1.0) < 1e-4
