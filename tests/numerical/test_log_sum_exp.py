"""Test log_sum_exp equivalence."""

import pytest

from cnake_charmer.cy.numerical.log_sum_exp import log_sum_exp as cy_log_sum_exp
from cnake_charmer.py.numerical.log_sum_exp import log_sum_exp as py_log_sum_exp


@pytest.mark.parametrize("n", [100, 1000, 50000])
def test_log_sum_exp_equivalence(n):
    py_result = py_log_sum_exp(n)
    cy_result = cy_log_sum_exp(n)

    # Check tuple structure
    assert isinstance(py_result, tuple), f"Python result should be tuple, got {type(py_result)}"
    assert isinstance(cy_result, tuple), f"Cython result should be tuple, got {type(cy_result)}"
    assert len(py_result) == 2
    assert len(cy_result) == 2

    # Float tolerance check
    for i in range(2):
        diff = abs(py_result[i] - cy_result[i]) / max(abs(py_result[i]), 1.0)
        assert diff < 1e-4, (
            f"Element {i} mismatch: py={py_result[i]}, cy={cy_result[i]}, rel_diff={diff}"
        )
