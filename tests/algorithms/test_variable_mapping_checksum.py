"""Test variable_mapping_checksum equivalence."""

import pytest

from cnake_charmer.cy.algorithms.variable_mapping_checksum import (
    variable_mapping_checksum as cy_func,
)
from cnake_charmer.py.algorithms.variable_mapping_checksum import (
    variable_mapping_checksum as py_func,
)


@pytest.mark.parametrize(
    "n_vars,seed,draws,clause_len",
    [
        (32, 7, 200, 3),
        (64, 1337, 700, 4),
        (80, 4242, 1200, 5),
    ],
)
def test_variable_mapping_checksum_equivalence(n_vars, seed, draws, clause_len):
    assert py_func(n_vars, seed, draws, clause_len) == cy_func(n_vars, seed, draws, clause_len)
