"""Test knuth_morris_pratt_all equivalence."""

import pytest

from cnake_charmer.cy.string_processing.knuth_morris_pratt_all import (
    knuth_morris_pratt_all as cy_func,
)
from cnake_charmer.py.string_processing.knuth_morris_pratt_all import (
    knuth_morris_pratt_all as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_knuth_morris_pratt_all_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
