"""Test conditional_event_prob equivalence."""

import pytest

from cnake_charmer.cy.algorithms.conditional_event_prob import conditional_event_prob as cy_func
from cnake_charmer.py.algorithms.conditional_event_prob import conditional_event_prob as py_func


@pytest.mark.parametrize(
    "window,rows,delta",
    [
        (3, 25, 1),
        (5, 200, 2),
        (7, 1000, 3),
        (9, 2500, 4),
    ],
)
def test_conditional_event_prob_equivalence(window, rows, delta):
    assert py_func(window, rows, delta) == cy_func(window, rows, delta)
