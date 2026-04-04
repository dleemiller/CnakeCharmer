"""Test queue_relay_state_class equivalence."""

import pytest

from cnake_data.cy.algorithms.queue_relay_state_class import queue_relay_state_class as cy_func
from cnake_data.py.algorithms.queue_relay_state_class import queue_relay_state_class as py_func


@pytest.mark.parametrize(
    "capacity,rounds,seed,mask", [(16, 400, 7, 255), (64, 1700, 13, 1023), (128, 2500, 99, 2047)]
)
def test_queue_relay_state_class_equivalence(capacity, rounds, seed, mask):
    assert py_func(capacity, rounds, seed, mask) == cy_func(capacity, rounds, seed, mask)
