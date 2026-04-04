"""Test rabin_karp equivalence."""

import pytest

from cnake_data.cy.string_processing.rabin_karp import rabin_karp as cy_func
from cnake_data.py.string_processing.rabin_karp import rabin_karp as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_rabin_karp_equivalence(n):
    assert py_func(n) == cy_func(n)
