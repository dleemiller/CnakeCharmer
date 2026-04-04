"""Test ring_queue_class_ops equivalence."""

import pytest

from cnake_data.cy.algorithms.ring_queue_class_ops import ring_queue_class_ops as cy_func
from cnake_data.py.algorithms.ring_queue_class_ops import ring_queue_class_ops as py_func


@pytest.mark.parametrize("capacity,rounds,seed", [(32, 100, 7), (64, 300, 19), (48, 250, 123)])
def test_ring_queue_class_ops_equivalence(capacity, rounds, seed):
    assert py_func(capacity, rounds, seed) == cy_func(capacity, rounds, seed)
