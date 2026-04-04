"""Test discrete_cosine_transform equivalence."""

import pytest

from cnake_data.cy.numerical.discrete_cosine_transform import (
    discrete_cosine_transform as cy_func,
)
from cnake_data.py.numerical.discrete_cosine_transform import (
    discrete_cosine_transform as py_func,
)


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_discrete_cosine_transform_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={p}, cy={c}"
