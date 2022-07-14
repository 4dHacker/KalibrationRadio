import numpy as np
import math
import nifty8 as ift
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator


def genPhaseMatrix(n):
    O=math.comb(n,2)
    mat=np.zeros(O,dtype=int)
    pos=[(x,y) for x in range(n) for y in range(x+1,n)]
    maxRank=math.comb(n-1,2)
    for i in range(0,n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                row=np.zeros(O,dtype=int)
                row[pos.index((i,j))]=1
                row[pos.index((i,k))]=-1
                row[pos.index((j,k))]=1

                newMat = np.vstack([mat,row])
                rnk= np.linalg.matrix_rank(newMat)
                if rnk>np.linalg.matrix_rank(mat):
                    mat=newMat
                    if rnk==maxRank:
                        return mat[1:,:]
    mat=mat[1:,:]
    dom=ift.UnstructuredDomain(mat.shape)
    fied=ift.Field(dom,mat)
    return mat


class VisPhasesToClosurePhases(ift.LinearOperator):
    def __init__(self, time, ant1, ant2):
        self._domain = ift.UnstructuredDomain([time.shape[0], 1])
        self._capability = self.TIMES | self.ADJOINT_TIMES

        assert np.all(np.diff(time) >= 0)

        rows, cols, data = [], [], []
        for tt in np.unique(time):
            print(time[time == tt])
            print(ant1[time == tt])
            print(ant2[time == tt])
            exit()
        #plt.plot(time)
        #plt.show()
        #print(times)
        exit()


        # FIXME data, rows, columns, matrix_shp, ausrechnen
        self._mat = coo_matrix((data, (rows, cols)), matrix_shp)
        self._mat = aslinearoperator(self._mat)

        self._target = ift.UnstructuredDomain(n_closure_phases)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = self._mat.matvec(np.squeeze(x.val)).reshape(self.target.shape)
        else:
            res = self._mat.rmatvec(np.squeeze(x.val)).reshape(self.domain.shape)
        return Field(self._tgt(mode), res)


def getClosurePhaseOperator(d):
    # vis -> vis-Phases
    # vis = |vis| * exp(i * phi)
    # phi = real(-i * log(vis / |vis|))

    vis = ift.Operator.identity_operator(d["vis"].domain)
    vis_ph = (vis * vis.abs().reciprocal()).log().scale(-1j).real

    VisPhasesToClosurePhases(d["time"], d["ant1"], d["ant2"])

    # Matrix-Multiplikation
    return []
