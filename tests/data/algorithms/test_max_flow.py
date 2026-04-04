"""Test max_flow equivalence."""

import pytest

from cnake_data.cy.algorithms.max_flow import max_flow as cy_max_flow
from cnake_data.py.algorithms.max_flow import max_flow as py_max_flow


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_max_flow_equivalence(n):
    py_result = py_max_flow(n)
    cy_result = cy_max_flow(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
