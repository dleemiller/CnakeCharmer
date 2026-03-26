"""Test majority_element_verify equivalence."""

import pytest

from cnake_charmer.cy.algorithms.majority_element_verify import (
    majority_element_verify as cy_majority_element_verify,
)
from cnake_charmer.py.algorithms.majority_element_verify import (
    majority_element_verify as py_majority_element_verify,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_majority_element_verify_equivalence(n):
    assert py_majority_element_verify(n) == cy_majority_element_verify(n)
