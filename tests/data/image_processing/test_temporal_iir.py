"""Test temporal_iir equivalence."""

import pytest

from cnake_data.cy.image_processing.temporal_iir import temporal_iir as cy_func
from cnake_data.py.image_processing.temporal_iir import temporal_iir as py_func


@pytest.mark.parametrize(
    "rows,cols,n_frames,alpha",
    [
        (40, 40, 15, 0.3),
        (80, 80, 30, 0.3),
        (60, 60, 20, 0.5),
        (50, 50, 10, 0.2),
    ],
)
def test_temporal_iir_equivalence(rows, cols, n_frames, alpha):
    assert py_func(rows, cols, n_frames, alpha) == cy_func(rows, cols, n_frames, alpha)
