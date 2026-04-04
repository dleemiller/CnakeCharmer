"""Test signal_distance equivalence."""

import pytest

from cnake_data.cy.numerical.signal_distance import signal_distance as cy_signal_distance
from cnake_data.py.numerical.signal_distance import signal_distance as py_signal_distance


@pytest.mark.parametrize("n", [1, 10, 100, 500])
def test_signal_distance_equivalence(n):
    py_result = py_signal_distance(n)
    cy_result = cy_signal_distance(n)
    for py_val, cy_val in zip(py_result, cy_result, strict=False):
        assert abs(py_val - cy_val) / max(abs(py_val), 1.0) < 1e-4
