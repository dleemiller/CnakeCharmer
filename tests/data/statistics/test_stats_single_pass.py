"""Test stats_single_pass equivalence."""

import pytest

from cnake_data.cy.statistics.stats_single_pass import stats_single_pass as cy_stats
from cnake_data.py.statistics.stats_single_pass import stats_single_pass as py_stats


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_stats_single_pass_equivalence(n):
    py_result = py_stats(n)
    cy_result = cy_stats(n)
    assert py_result["len"] == cy_result["len"]
    assert abs(py_result["min"] - cy_result["min"]) < 1e-6
    assert abs(py_result["max"] - cy_result["max"]) < 1e-6
    assert abs(py_result["sum"] - cy_result["sum"]) < 1e-3
    assert abs(py_result["mean"] - cy_result["mean"]) < 1e-6
    assert abs(py_result["pstdev"] - cy_result["pstdev"]) < 1e-3
