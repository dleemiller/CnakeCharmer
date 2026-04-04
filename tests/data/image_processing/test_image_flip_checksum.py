"""Test image_flip_checksum equivalence."""

import pytest

from cnake_data.cy.image_processing.image_flip_checksum import image_flip_checksum as cy_func
from cnake_data.py.image_processing.image_flip_checksum import image_flip_checksum as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_image_flip_checksum_equivalence(n):
    assert py_func(n) == cy_func(n)
