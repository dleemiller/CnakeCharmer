"""Test simulated_annealing_deterministic equivalence."""

import pytest

from cnake_data.cy.optimization.simulated_annealing_deterministic import (
    simulated_annealing_deterministic as cy_func,
)
from cnake_data.py.optimization.simulated_annealing_deterministic import (
    simulated_annealing_deterministic as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_simulated_annealing_deterministic_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert isinstance(py_result, tuple) and len(py_result) == 3
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, (
            f"Mismatch at n={n}: py={py_result}, cy={cy_result}"
        )
