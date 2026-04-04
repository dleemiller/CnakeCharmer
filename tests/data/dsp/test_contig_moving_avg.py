"""Test moving average equivalence."""

import pytest

from cnake_data.cy.dsp.contig_moving_avg import (
    contig_moving_avg as cy_contig_moving_avg,
)
from cnake_data.py.dsp.contig_moving_avg import (
    contig_moving_avg as py_contig_moving_avg,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_contig_moving_avg_equivalence(n):
    py_result = py_contig_moving_avg(n)
    cy_result = cy_contig_moving_avg(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
