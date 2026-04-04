"""Test simulated_annealing_tsp equivalence."""

import pytest

from cnake_data.cy.optimization.simulated_annealing_tsp import simulated_annealing_tsp as cy_func
from cnake_data.py.optimization.simulated_annealing_tsp import simulated_annealing_tsp as py_func


@pytest.mark.parametrize("n", [10, 30, 50, 100])
def test_simulated_annealing_tsp_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # Distance comparison with relative tolerance
    assert abs(py_result[0] - cy_result[0]) / max(abs(py_result[0]), 1.0) < 1e-4, (
        f"Distance mismatch: py={py_result[0]}, cy={cy_result[0]}"
    )
    # City indices must match exactly
    assert py_result[1] == cy_result[1], (
        f"First city mismatch: py={py_result[1]}, cy={cy_result[1]}"
    )
    assert py_result[2] == cy_result[2], f"Mid city mismatch: py={py_result[2]}, cy={cy_result[2]}"
