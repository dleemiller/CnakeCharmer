"""Test manacher equivalence."""

import pytest

from cnake_data.cy.string_processing.manacher import manacher as cy_func
from cnake_data.py.string_processing.manacher import manacher as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 500, 1000])
def test_manacher_equivalence(n):
    assert py_func(n) == cy_func(n)
