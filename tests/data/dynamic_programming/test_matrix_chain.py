"""Test matrix_chain equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.matrix_chain import matrix_chain as cy_matrix_chain
from cnake_data.py.dynamic_programming.matrix_chain import matrix_chain as py_matrix_chain


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_matrix_chain_equivalence(n):
    assert py_matrix_chain(n) == cy_matrix_chain(n)
