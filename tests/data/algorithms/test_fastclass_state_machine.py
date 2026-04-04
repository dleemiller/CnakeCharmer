"""Test fastclass_state_machine equivalence."""

import pytest

from cnake_data.cy.algorithms.fastclass_state_machine import (
    fastclass_state_machine as cy_func,
)
from cnake_data.py.algorithms.fastclass_state_machine import (
    fastclass_state_machine as py_func,
)


@pytest.mark.parametrize(
    "n_objs,steps,seed,mask",
    [(8, 300, 7, 255), (16, 1000, 1337, 1023), (12, 900, 99, 511)],
)
def test_fastclass_state_machine_equivalence(n_objs, steps, seed, mask):
    assert py_func(n_objs, steps, seed, mask) == cy_func(n_objs, steps, seed, mask)
