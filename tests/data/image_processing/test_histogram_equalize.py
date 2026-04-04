"""Test histogram_equalize equivalence."""

import pytest

from cnake_data.cy.image_processing.histogram_equalize import (
    histogram_equalize as cy_histogram_equalize,
)
from cnake_data.py.image_processing.histogram_equalize import (
    histogram_equalize as py_histogram_equalize,
)


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_histogram_equalize_equivalence(n):
    py_result = py_histogram_equalize(n)
    cy_result = cy_histogram_equalize(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
