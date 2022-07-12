import math as m
import pytest as pt
import numpy as np

def vis_design_matrix(n):
    if n == 2: return np.array([1, -1])
    tl = np.ones(shape=(n-1, 1))
    tr = np.identity(n-1) * -1
    bl = np.zeros(shape=(m.comb(n, 2)-(n-1), 1))
    br = vis_design_matrix(n-1)
    return np.block([[tl, tr], [bl, br]])

@pt.mark.parametrize('n_ant', [4])
def test_closure_phases(n_ant):
    clo_ph = closure_phase_design_matrix(n_ant)
    vis = vis_design_matrix(n_ant)
    np.testing.assert_allclose(clo_ph @ vis, 0, rtol=1e-13, atol=1e-14)
