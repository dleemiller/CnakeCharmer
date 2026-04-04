"""Test token_shingle_distance equivalence."""

import pytest

from cnake_charmer.cy.string_processing.token_shingle_distance import (
    token_shingle_distance as cy_func,
)
from cnake_charmer.py.string_processing.token_shingle_distance import (
    token_shingle_distance as py_func,
)


@pytest.mark.parametrize("args", [(5000, 3, 26, 5), (9000, 4, 32, 11), (12000, 5, 40, 17)])
def test_token_shingle_distance_equivalence(args):
    p = py_func(*args)
    c = cy_func(*args)
    for pv, cv in zip(p, c, strict=False):
        assert abs(pv - cv) / max(abs(pv), 1.0) < 1e-8
