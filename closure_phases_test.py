import math as m
import pytest as pt
import numpy as np
import nifty8 as ift


def vis_design_matrix(n):
    if n == 2: return np.array([1, -1])
    tl = np.ones(shape=(n-1, 1))
    tr = np.identity(n-1) * -1
    bl = np.zeros(shape=(m.comb(n, 2)-(n-1), 1))
    br = vis_design_matrix(n-1)

    return np.block([[tl, tr], [bl, br]])


def closure_phase_design_matrix(n_ant):
    file_field = open(f'PhaseMatrixes1-37/PhaseMatrix{n_ant}.npy', 'rb')
    field = np.load(file_field)
    if n_ant==3:
        ma = np.zeros((1, 3), dtype=int)
        field = np.vstack((field, ma))[0, :]
    realField=ift.Field()
    return field


@pt.mark.parametrize('n_ant', [3])
def test_closure_phases(n_ant):
    clo_ph = closure_phase_design_matrix(n_ant)
    vis = vis_design_matrix(n_ant)
    np.testing.assert_allclose(clo_ph @ vis, 0, rtol=1e-13, atol=1e-14)

