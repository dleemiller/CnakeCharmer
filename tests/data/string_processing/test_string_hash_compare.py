"""Test string_hash_compare equivalence."""

import pytest

from cnake_data.cy.string_processing.string_hash_compare import string_hash_compare as cy_func
from cnake_data.py.string_processing.string_hash_compare import string_hash_compare as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000, 5000])
def test_string_hash_compare_equivalence(n):
    assert py_func(n) == cy_func(n)
