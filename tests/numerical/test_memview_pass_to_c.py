"""Test memoryview pass-to-C equivalence."""

import pytest

from cnake_charmer.cy.numerical.memview_pass_to_c import (
    memview_pass_to_c as cy_memview_pass_to_c,
)
from cnake_charmer.py.numerical.memview_pass_to_c import (
    memview_pass_to_c as py_memview_pass_to_c,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_memview_pass_to_c_equivalence(n):
    py_result = py_memview_pass_to_c(n)
    cy_result = cy_memview_pass_to_c(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
