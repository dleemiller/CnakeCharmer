"""Test bicluster_scs_score equivalence."""

import pytest

from cnake_data.cy.statistics.bicluster_scs_score import bicluster_scs_score as cy_func
from cnake_data.py.statistics.bicluster_scs_score import bicluster_scs_score as py_func


@pytest.mark.parametrize("args", [(20, 18, 7), (36, 24, 13), (44, 30, 19)])
def test_bicluster_scs_score_equivalence(args):
    p = py_func(*args)
    c = cy_func(*args)
    for pv, cv in zip(p, c, strict=False):
        assert abs(pv - cv) / max(abs(pv), 1.0) < 1e-8
