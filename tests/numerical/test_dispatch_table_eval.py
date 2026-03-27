"""Test dispatch_table_eval equivalence."""

import pytest

from cnake_charmer.cy.numerical.dispatch_table_eval import (
    dispatch_table_eval as cy_dispatch_table_eval,
)
from cnake_charmer.py.numerical.dispatch_table_eval import (
    dispatch_table_eval as py_dispatch_table_eval,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_dispatch_table_eval_equivalence(n):
    py_result = py_dispatch_table_eval(n)
    cy_result = cy_dispatch_table_eval(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
