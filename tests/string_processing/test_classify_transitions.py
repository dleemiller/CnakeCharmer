"""Test classify_transitions equivalence."""

import pytest

from cnake_charmer.cy.string_processing.classify_transitions import (
    classify_transitions as cy_func,
)
from cnake_charmer.py.string_processing.classify_transitions import (
    classify_transitions as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_classify_transitions_equivalence(n):
    assert py_func(n) == cy_func(n)
