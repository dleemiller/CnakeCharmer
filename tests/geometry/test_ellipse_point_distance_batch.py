"""Test ellipse_point_distance_batch equivalence."""

import pytest

from cnake_charmer.cy.geometry.ellipse_point_distance_batch import (
    ellipse_point_distance_batch as cy_func,
)
from cnake_charmer.py.geometry.ellipse_point_distance_batch import (
    ellipse_point_distance_batch as py_func,
)


@pytest.mark.parametrize("args", [(4.0, 2.0, 3000, 5), (6.0, 3.0, 5000, 11), (9.0, 4.5, 8000, 17)])
def test_ellipse_point_distance_batch_equivalence(args):
    p = py_func(*args)
    c = cy_func(*args)
    assert p[2] == c[2]
    for i in (0, 1):
        assert abs(p[i] - c[i]) / max(abs(p[i]), 1.0) < 1e-8
