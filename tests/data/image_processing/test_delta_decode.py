"""Test delta_decode equivalence."""

import pytest

from cnake_data.cy.image_processing.delta_decode import delta_decode as cy_func
from cnake_data.py.image_processing.delta_decode import delta_decode as py_func


@pytest.mark.parametrize("n", [50, 200, 500, 1000])
def test_delta_decode_equivalence(n):
    assert py_func(n) == cy_func(n)
