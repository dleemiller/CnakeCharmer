"""Test huber_loss equivalence."""

import pytest

from cnake_data.cy.statistics.huber_loss import huber_loss as cy_func
from cnake_data.py.statistics.huber_loss import huber_loss as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_huber_loss_equivalence(n):
    py_loss, py_max = py_func(n)
    cy_loss, cy_max = cy_func(n)
    assert abs(py_loss - cy_loss) / max(abs(py_loss), 1.0) < 1e-4
    assert abs(py_max - cy_max) / max(abs(py_max), 1.0) < 1e-4
