"""Test extern_string_ops equivalence."""

import pytest

from cnake_charmer.cy.string_processing.extern_string_ops import (
    extern_string_ops as cy_func,
)
from cnake_charmer.py.string_processing.extern_string_ops import (
    extern_string_ops as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_extern_string_ops_equivalence(n):
    assert py_func(n) == cy_func(n)
