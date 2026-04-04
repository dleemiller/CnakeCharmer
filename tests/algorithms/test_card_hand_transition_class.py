"""Test card_hand_transition_class equivalence."""

import pytest

from cnake_charmer.cy.algorithms.card_hand_transition_class import (
    card_hand_transition_class as cy_func,
)
from cnake_charmer.py.algorithms.card_hand_transition_class import (
    card_hand_transition_class as py_func,
)


@pytest.mark.parametrize(
    "n_hands,seed,max_value,n_suits", [(100, 7, 9, 4), (1500, 13, 13, 4), (2400, 21, 12, 5)]
)
def test_card_hand_transition_class_equivalence(n_hands, seed, max_value, n_suits):
    assert py_func(n_hands, seed, max_value, n_suits) == cy_func(n_hands, seed, max_value, n_suits)
