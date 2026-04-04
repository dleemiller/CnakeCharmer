"""Test signature_l2 equivalence."""

import pytest

from cnake_charmer.cy.numerical.signature_l2 import signature_l2 as cy_func
from cnake_charmer.py.numerical.signature_l2 import signature_l2 as py_func


@pytest.mark.parametrize(
    "n,stride,bias",
    [
        (10, 3, 0.1),
        (128, 7, 0.35),
        (1000, 11, -0.2),
        (4096, 17, 0.0),
    ],
)
def test_signature_l2_equivalence(n, stride, bias):
    assert py_func(n, stride, bias) == cy_func(n, stride, bias)
