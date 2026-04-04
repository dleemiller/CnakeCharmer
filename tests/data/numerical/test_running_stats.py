"""Test running_stats equivalence."""

import pytest

from cnake_data.cy.numerical.running_stats import running_stats as cy_running_stats
from cnake_data.py.numerical.running_stats import running_stats as py_running_stats


@pytest.mark.parametrize("n", [500, 5000, 50000, 200000])
def test_running_stats_equivalence(n):
    py_result = py_running_stats(n)
    cy_result = cy_running_stats(n)
    assert py_result == cy_result, f"Mismatch at n={n}: py={py_result}, cy={cy_result}"
