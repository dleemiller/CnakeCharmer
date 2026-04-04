"""Test strongly_connected equivalence."""

import pytest

from cnake_data.cy.graph.strongly_connected import strongly_connected as cy_func
from cnake_data.py.graph.strongly_connected import strongly_connected as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_strongly_connected_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
