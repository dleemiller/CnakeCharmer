"""Test gaussian_blur equivalence."""

import pytest

from cnake_data.cy.image_processing.gaussian_blur import gaussian_blur as cy_func
from cnake_data.py.image_processing.gaussian_blur import gaussian_blur as py_func


@pytest.mark.parametrize("n", [10, 20, 50, 100])
def test_gaussian_blur_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
